# Design notes: serverless SFT MLOps pipeline

An MLOps pipeline, deployed with AWS CDK, that fine-tunes an open-weight LLM using
**Amazon SageMaker AI serverless model customization** (supervised fine-tuning / SFT)
and wraps it with data preparation, evaluation, a quality gate, and model registration
inside a **SageMaker Pipeline**.

Research and design notes behind this project, kept from the planning phase. Section 4
records what was verified while implementing. For how to use the code, see
[../README.md](../README.md).

Differences from the plan below: the target model is Gemma 4 E4B-IT; data prep runs as a
`@step` rather than a `ProcessingStep`; deployment is a separate, approval-gated pipeline
that targets a SageMaker endpoint, because Bedrock Custom Model Import doesn't support the
Gemma 4 architecture.

---

## 1. Research summary

### 1.1 What "serverless model customization" is

- A fully managed SageMaker AI capability for customizing foundation models without
  choosing or managing GPU instances. SageMaker picks the hardware (P5 / P4de / P4d / G5),
  applies a pre-optimized recipe for the chosen technique, streams live metrics/logs,
  and tears down compute when the job completes.
- Supported techniques: **SFT**, **DPO**, **RLVR**, **RLAIF**, and multi-turn RL.
  This project uses SFT (LoRA).
- Training output is a **logged model in a Model Package Group**, not just a
  `model.tar.gz` in S3. Lineage (datasets -> training job -> model -> evaluation ->
  deployment) is captured automatically.
- Custom open-weight models can be deployed to **Amazon Bedrock** (serverless inference)
  or to a **SageMaker endpoint** (dedicated capacity).
- Source: https://docs.aws.amazon.com/sagemaker/latest/dg/customize-model.html

### 1.2 How it maps onto the SageMaker API (why Pipelines works)

Serverless customization jobs are ordinary **SageMaker Training Jobs**. `CreateTrainingJob`
accepts a `ServerlessJobConfig` in place of `ResourceConfig`:

| Field                    | Required | Values / notes                                                    |
|--------------------------|----------|-------------------------------------------------------------------|
| `BaseModelArn`           | yes      | ARN of a model in the SageMaker Public Hub (latest version used)  |
| `JobType`                | yes      | `FineTuning` \| `Evaluation`                                      |
| `CustomizationTechnique` | no       | `SFT` \| `DPO` \| `RLVR` \| `RLAIF`                               |
| `Peft`                   | no       | `LORA`                                                            |
| `SequenceLength`         | no       | `1K` .. `1024K` (e.g. `4K`); auto-selected if omitted             |
| `AcceptEula`             | no       | must be `true` for gated models                                   |
| `EvaluationType`         | no       | `LLMAJEvaluation` \| `CustomScorerEvaluation` \| `BenchmarkEvaluation` (for `JobType=Evaluation`) |
| `EvaluatorArn`           | no       | reward function / reward prompt ARN                               |

Because the unit of work is a training job, anything that can launch a training job
(SageMaker Pipelines, Step Functions, a Lambda, the Python SDK) can orchestrate it.
Note that **evaluation is also a serverless training job** (`JobType=Evaluation`), which
lets the eval step use the same mechanism as the SFT step.

Source: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ServerlessJobConfig.html

### 1.3 Python SDK v3 surface

`SFTTrainer` is the client-side helper. Omitting `compute=` selects serverless capacity.

```python
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

trainer = SFTTrainer(
    model="huggingface-reasoning-qwen3-8b",   # Public Hub model id
    training_type=TrainingType.LORA,
    model_package_group=MODEL_PACKAGE_GROUP,
    training_dataset=TRAINING_DATASET_ARN,    # dataset asset ARN or s3:// URI
    validation_dataset=VALIDATION_DATASET_ARN,
    sequence_length="4K",
    s3_output_path=S3_OUTPUT_PATH,
    role=ROLE_ARN,
    # no compute= -> serverless model customization
)
trainer.hyperparameters.max_epochs = 3
trainer.hyperparameters.global_batch_size = 8
trainer.hyperparameters.lora_rank = 16
trainer.hyperparameters.merge_weights = True

job = trainer.train(wait=True)
model_package_arn = job.output_model_package_arn
```

Source: https://aws.amazon.com/blogs/machine-learning/build-an-ai-powered-product-tagging-system-with-amazon-sagemaker-serverless-model-customization/

**Import path caveat.** The AWS docs page on job submission shows a different import
path and kwarg name than the blog:

```python
# docs: model-customize-open-weight-job.html (DPO example, same shape for SFT)
from sagemaker.modules.train.dpo_trainer import DPOTrainer
from sagemaker.modules.train.common import TrainingType

trainer = DPOTrainer(
    model=BASE_MODEL,
    training_type=TrainingType.LORA,
    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,   # vs model_package_group= in the blog
    training_dataset=TRAINING_DATASET,
    s3_output_path=S3_OUTPUT_PATH,
    sagemaker_session=sagemaker_session,
    role=ROLE_ARN,
)
```

`sagemaker.train.*` (blog) vs `sagemaker.modules.train.*` (docs) and
`model_package_group` vs `model_package_group_name` — one of these is likely an alias or
an older/newer SDK layout. Resolve against the installed SDK before writing `sft.py`.

Job lifecycle per the docs: submit -> training-job page (live metrics/logs) -> on
completion a **Custom Model** details page with artifacts location, hyperparameters,
logs, plus actions to (a) launch an evaluation job on another dataset and (b) deploy to
a SageMaker AI endpoint or Bedrock. Our pipeline reproduces (a) and (b) programmatically.

Source: https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-job.html

### 1.4 Customization assets: datasets and evaluators

Serverless customization uses first-class **assets** stored in the SageMaker AI
registry (hub):

- **Datasets** — JSONL, one example per line. SFT uses a conversation / labeled
  input-output format; RLVR uses prompt + ground-truth. Studio UI
  (Assets -> Datasets -> Upload Dataset) shows the required format per technique.
- **Evaluators** — **reward functions** (code, e.g. a Lambda or a local `.py` that the SDK
  packages) used for RLVR and custom-scorer evaluation, and **reward prompts** (natural
  language rubrics) used for RLAIF and LLM-as-a-judge evaluation.

SDK equivalent (what our data-prep step will call):

```python
from sagemaker.ai_registry.air_constants import REWARD_FUNCTION, REWARD_PROMPT
from sagemaker.ai_registry.dataset import DataSet, CustomizationTechnique
from sagemaker.ai_registry.evaluator import Evaluator

dataset = DataSet.create(
    name="sft-train-v1",
    source="s3://<bucket>/datasets/train.jsonl",   # or a local path
    customization_technique=CustomizationTechnique.SFT,
)
dataset.refresh()

# Reward function backed by an existing Lambda
evaluator = Evaluator.create(
    name="custom-scorer",
    source="arn:aws:lambda:<region>:<account>:function:<name>",
    type=REWARD_FUNCTION,
)
# ...or bring-your-own code; SDK packages it
evaluator = Evaluator.create(
    name="custom-scorer",
    source="./evaluators/score.py",
    type=REWARD_FUNCTION,
)
evaluator.wait()
```

Source: https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-create-assets-ui.html

### 1.5 Deploying the custom model

The deployment doc page itself is thin — it only states that from the custom model
details page you can deploy to **SageMaker AI Inference endpoints** or **Amazon Bedrock**.
Programmatic equivalents we will use in the (optional) deploy step:

| Target                | How                                                                 | When to pick                                   |
|-----------------------|---------------------------------------------------------------------|------------------------------------------------|
| SageMaker endpoint    | `CreateModel` from the model package ARN -> `CreateEndpointConfig` -> `CreateEndpoint` (or SDK v3 `ModelBuilder` / model package `.deploy()`) | dedicated capacity, control over throughput and tail latency |
| Bedrock Custom Model Import | `bedrock.create_model_import_job` pointing at the merged safetensors S3 prefix (`s3://.../output/model/`), then `create_custom_model_deployment` for on-demand inference; invoke via `bedrock-runtime` with the deployment/model ARN as `modelId` | serverless inference, scale-to-zero, pay per use |

Notes:
- Bedrock import needs merged full weights with config + tokenizer at the top level of
  the S3 prefix — set `trainer.hyperparameters.merge_weights = True` on the SFT job.
- Bedrock import requires a Bedrock service role that can read the output bucket, plus
  `bedrock:CreateModelImportJob`, `GetModelImportJob`, `CreateCustomModelDeployment`,
  `GetCustomModelDeployment` on the pipeline/deploy role.
- Bedrock Custom Model Import region support is narrower than SageMaker's; confirm the
  stack region supports both before committing to the Bedrock path.
- Deployment lineage (model package -> endpoint / Bedrock model) is tracked by the
  Model Registry when done from the model package.

Sources:
https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-deployment.html,
https://docs.aws.amazon.com/bedrock/latest/userguide/create-custom-model-sdks.html,
https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-custom-model-import-now-generally-available/

### 1.6 Using it from SageMaker Pipelines

Two options:

1. **`@step` decorator (recommended, SDK-version independent).** Wrap `SFTTrainer.train()`
   (or `DataSet.create()`, or an evaluation call) in a Python function decorated with
   `@step`. Each `@step` runs in its own small training job and submits/waits on the
   serverless job. Return values (e.g. the model package ARN) flow to downstream steps
   as pipeline properties.
2. **Native `TrainingStep` with `step_args` from the trainer.** Shows the serverless job
   as the step's own training job in the Studio DAG, but depends on whether the
   installed `sagemaker` v3 `TrainingStep` serializes `ServerlessJobConfig`.
   **Unverified** — check once the SDK is installed in a venv:

   ```bash
   python -c "import sagemaker.workflow.steps as s, inspect; print('ServerlessJobConfig' in inspect.getsource(s))"
   ```

Trade-off with option 1: the `@step` instance blocks on `wait=True` for the duration of
the serverless job. Mitigation if cost matters: split into a submit step that returns the
job name and a downstream step that polls `DescribeTrainingJob` (or use a
`CallbackStep` + Lambda/EventBridge).

### 1.7 Constraints and gotchas

- **Regions**: serverless model customization was reported available in `us-east-1`,
  `us-west-2`, `ap-northeast-1`, `eu-west-1` (from earlier research; re-verify against
  the current docs before pinning the CDK stack region).
- **IAM**: pipeline execution role needs `sagemaker:CreateTrainingJob` /
  `DescribeTrainingJob`, `sagemaker:DescribeHubContent` on the Public Hub base model,
  model package group read/write, hub dataset/evaluator asset permissions, S3 for datasets
  and outputs, CloudWatch Logs, and (if evaluation uses Bedrock Evaluations) the
  relevant `bedrock:*` evaluation permissions.
- **EULA**: gated base models require `AcceptEula=true`.
- **Downstream contract**: consume `output_model_package_arn`, not
  `ModelArtifacts.S3ModelArtifacts`.
- **Bedrock Evaluations data movement**: LLM-as-a-judge evaluation may process data
  across Regions within the same geography — check compliance requirements.
- **Training session APIs** (`forward_backward()` etc.) are a different, stateful
  interaction model; out of scope for this pipeline.

---

## 2. Target architecture

```
                 ┌──────────────────────── CDK stack ────────────────────────┐
                 │  S3 (raw / curated / outputs)   IAM roles   Model Pkg Grp  │
                 │  SageMaker Pipeline definition  EventBridge  SNS/alerts    │
                 └───────────────────────────────────────────────────────────┘

 raw data (S3) ──► [1] Data prep ──► [2] Register datasets ──► [3] Serverless SFT
                   ProcessingStep      @step DataSet.create     @step SFTTrainer (LoRA)
                   validate/split      (train + validation)     -> model package ARN
                   write JSONL
                                                                      │
                   [6] Register / approve ◄── [5] Condition ◄── [4] Evaluate
                   ModelStep /                 metric >= threshold   serverless Evaluation job
                   update approval status      else FAIL step         (LLM-as-a-judge or
                                                                      custom scorer evaluator)
                                                                      and/or baseline compare
                             │
                             ▼
                   [7] (optional) Deploy to Bedrock or SageMaker endpoint
```

### Pipeline steps

| # | Step                 | Mechanism                                   | Output                                   |
|---|----------------------|---------------------------------------------|------------------------------------------|
| 1 | Data prep            | `ProcessingStep` (sklearn/script container)  | `train.jsonl`, `validation.jsonl`, `eval.jsonl` in S3 |
| 2 | Register datasets    | `@step` -> `DataSet.create(...)`             | dataset asset ARNs                        |
| 3 | Serverless SFT       | `@step` -> `SFTTrainer(...).train()`         | `output_model_package_arn`                |
| 4 | Evaluate             | `@step` -> serverless `Evaluation` job (LLM-as-a-judge / custom scorer); optionally same eval on base model for a delta | metrics JSON in S3 |
| 5 | Quality gate         | `ConditionStep` on metric                    | pass -> step 6, fail -> `FailStep`        |
| 6 | Register / approve   | update model package approval status         | `Approved` model package version          |
| 7 | Deploy (optional)    | Bedrock custom model import or SageMaker endpoint | inference endpoint / model ID        |

### CDK responsibilities

- S3 buckets (raw, curated, pipeline artifacts) with lifecycle rules and SSE.
- Pipeline execution role (least privilege, scoped to the base model hub ARN,
  the model package group, and the project buckets).
- Model Package Group.
- `AWS::SageMaker::Pipeline` resource, with the pipeline definition generated by the
  Python SDK and committed as JSON (or built at synth time).
- EventBridge rule on pipeline execution state changes -> SNS for notifications.
- Optional: EventBridge schedule or S3 event trigger to kick off executions.

---

## 3. Repo layout

See [Repository layout](../README.md#repository-layout) in the README.

---

## 4. Open questions / to verify

Answered while implementing (SDK: sagemaker-train 1.14, sagemaker-mlops 1.12,
sagemaker-core 2.23 in SageMaker Distribution 4.6):

- [x] Region: `us-west-2` (Oregon) supports serverless customization; the other listed
      Regions are `us-east-1`, `ap-northeast-1` and `eu-west-1`.
- [x] `TrainingStep`: not usable. `SFTTrainer.train()` calls `TrainingJob.create()`
      directly and isn't pipeline-session aware, so it can't produce `step_args`. The
      pipeline uses option 1 (`@step`), and `ServerlessSFT` waits on the job.
- [x] Base model: `huggingface-vlm-gemma-4-e4b-it` (hub version 1.8.1, Apache-2.0, no
      EULA). It supports SFT, DPO, RLVR and RLAIF, with LoRA and full fine-tuning. There is
      no `sequence_length` argument; use the `dataset_max_len` hyperparameter (256-32768,
      default 32768). Recipe constraints: `global_batch_size` must be one of
      128/256/512/1024, so the dataset needs at least 128 examples. `lora_rank` is one of
      8/16/32/64/128 and `lora_alpha` one of 16/32/64/128/256.
- [x] Gating metric: the Gemma 4 SFT recipe logs only `train/*` metrics to MLflow. It logs
      no validation loss, even though the validation channel is mounted. The gate
      currently checks the final `train/loss` (first run: 2.23 -> 1.17 over 8 steps).
- [x] SDK paths: `sagemaker.train.*` and `model_package_group=` are correct.
      `sagemaker.modules` and `sagemaker.workflow` were removed in v3; pipelines live in
      `sagemaker.mlops.workflow` and `sagemaker.core.workflow`.
- [x] IAM: the stack role uses `AmazonSageMakerModelCustomizationCoreAccess` plus S3 on
      the data and artifact buckets, `sagemaker:CreateHub`/`DescribeHub`/`AddTags`/`ListTags`,
      and MLflow metric reads. That managed policy only passes roles under `service-role/`
      whose name contains `SageMaker`, and only reaches buckets whose name contains
      `sagemaker`, so the stack names its role and bucket to match.

Still open:

- [ ] Replace the convergence gate with a held-out evaluation, for example a serverless
      `LLMAsJudgeEvaluator` or `BenchMarkEvaluator` job on `eval.jsonl`, which data prep
      already writes. Gate on its score and compare against the base model.
- [x] Deployment target: a separate deploy pipeline with manual approval, deploying to a
      SageMaker real-time endpoint. Bedrock Custom Model Import doesn't support the Gemma 4
      architecture (`Gemma4ForConditionalGeneration`); its supported list is Mistral,
      Mixtral, Flan-T5, Llama 2/3.x/Mllama, GPTBigCode, Qwen2/2.5/2-VL/2.5-VL/3 and GPT-OSS.
      The SFT job writes merged weights to `checkpoints/hf_merged/` (17 GB for Gemma 4 E4B)
      as well as the LoRA adapter in `checkpoints/hf/`, so a supported model could be
      imported from `hf_merged/`.
- [ ] `Evaluator.create` IAM (only needed for RLVR / custom-scorer evaluation).

## 5. References

- Serverless model customization overview:
  https://docs.aws.amazon.com/sagemaker/latest/dg/customize-model.html
- Creating assets (datasets / evaluators):
  https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-create-assets-ui.html
- Job submission (UI + SDK) and custom model lifecycle:
  https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-job.html
- Model deployment (SageMaker endpoint / Bedrock):
  https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-deployment.html
- Bedrock Custom Model Import (SDK):
  https://docs.aws.amazon.com/bedrock/latest/userguide/create-custom-model-sdks.html
- `ServerlessJobConfig` API reference:
  https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ServerlessJobConfig.html
- Blog: product tagging with serverless SFT + RLVR (SDK v3 `SFTTrainer` example):
  https://aws.amazon.com/blogs/machine-learning/build-an-ai-powered-product-tagging-system-with-amazon-sagemaker-serverless-model-customization/
- Blog: fine-tune Nemotron 3 with serverless customization (data prep guidance):
  https://aws.amazon.com/blogs/machine-learning/fine-tune-nvidia-nemotron-3-models-with-amazon-sagemaker-ai-serverless-model-customization/
- Blog: training session APIs (positioning vs serverless managed APIs):
  https://aws.amazon.com/blogs/machine-learning/write-your-own-training-loop-without-managing-gpus-with-amazon-sagemaker-training-sessions/
