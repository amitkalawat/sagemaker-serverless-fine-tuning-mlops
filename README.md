# Serverless SFT MLOps pipelines for Gemma 4 (SageMaker + CDK)

Fine-tunes **Gemma 4 E4B-IT** (`huggingface-vlm-gemma-4-e4b-it`, the smallest Gemma 4 model
with SageMaker customization support) with **SageMaker AI serverless model customization**
(SFT + LoRA, no GPU instances to manage), then deploys approved models to a **SageMaker
real-time endpoint**. Two SageMaker Pipelines, defined with the SageMaker Python SDK v3 and
deployed with AWS CDK:

```
 Training pipeline (gemma4-e4b-serverless-sft-cdk)
   PrepareData -> RegisterDatasets -> ServerlessSFT -> Evaluate -> QualityGate
                                                                    |-- pass: RegisterCandidate
                                                                    |         (model package -> PendingManualApproval)
                                                                    `-- fail: QualityGateFailed
        |
        | EventBridge: training execution Succeeded
        v
 Deploy pipeline (gemma4-e4b-sft-deploy-cdk)
   ResolveModel -> ManualApproval -> ApproveModelPackage -> DeployEndpoint -> SmokeTest
   (SNS request)   (SQS callback,    (package -> Approved)  (base + LoRA       (held-out prompts,
                    approve/reject                           adapter inference  old adapters
                    script)                                  components)        removed)
```

- **Training and deployment are separate.** Training only produces *candidates*. Nothing
  reaches the endpoint without a human approval in the deploy pipeline.
- **Everything is in CDK:** the pipelines, the IAM role, the bucket, the model registry
  group, the approval queue, the notifications and the event trigger.
- **Everything is tracked in MLflow** (the Studio domain's MLflow app): params, data stats,
  training curves, the gate decision, the approver, smoke-test outputs and every job's logs.

Training data: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(CC BY-SA 3.0), downloaded at run time from a pinned revision; it isn't included in this
repository. Design notes, research and verified API details: [docs/DESIGN.md](docs/DESIGN.md).

## Contents

1. [Repository layout](#repository-layout)
2. [Prerequisites](#prerequisites)
3. [First-time setup](#first-time-setup)
4. [Runbook: run a training and deploy cycle](#runbook-run-a-training-and-deploy-cycle)
5. [Getting a stronger fine-tune](#getting-a-stronger-fine-tune)
6. [What to look at in MLflow](#what-to-look-at-in-mlflow)
7. [Configuration](#configuration)
8. [AWS resources](#aws-resources)
9. [Iterating without CDK](#iterating-without-cdk)
10. [Cost and cleanup](#cost-and-cleanup)
11. [Troubleshooting](#troubleshooting)
12. [Test record](#test-record)
13. [Security](#security)
14. [Known limitations](#known-limitations)

## Repository layout

```
.
├── README.md                     # this file
├── docs/DESIGN.md                # research, design decisions, verified SDK/API facts
├── .gitignore                    # keeps local config, runtime output and CDK build files out
├── config/
│   ├── pipeline.yaml             # THE configuration: model, data, hyperparameters, gate,
│   │                             #   MLflow, deploy settings, CDK resource names
│   └── pipeline.local.example.yaml   # template for your gitignored pipeline.local.yaml
├── pipeline/
│   ├── config.py                 # loads config/pipeline.yaml
│   ├── definition.py             # TRAINING pipeline: @step functions + build_pipeline()
│   ├── deploy_definition.py      # DEPLOY pipeline: @step functions + CallbackStep approval
│   └── steps/                    # plain functions used by the pipelines and the scripts
│       ├── data_prep.py          # Dolly 15k -> {"prompt","completion"} JSONL splits in S3
│       ├── sft.py                # model package group, dataset registration, SFTTrainer
│       ├── deploy.py             # resolve package, ModelBuilder deploy, smoke test, cleanup
│       └── tracking.py           # MLflow runs, metric copying, CloudWatch logs as artifacts
├── scripts/
│   ├── status.py                 # where things stand: executions, approvals, versions, endpoint
│   ├── approve_deployment.py     # approve/reject a deploy execution waiting for approval
│   ├── test_endpoint.py          # fine-tuned vs base model on held-out data, or ad-hoc prompts
│   └── run_sft.py                # run the training steps in-process from Studio (no pipeline)
└── cdk/
    ├── cdk.sh                    # CDK CLI wrapper: venv + Studio jsii workaround
    ├── app.py                    # builds both pipeline definitions at synth time
    ├── cdk.json, requirements.txt
    └── stacks/sft_pipeline_stack.py   # all AWS resources (see AWS resources)
```

Generated at runtime (not source, safe to delete): `data/` (local copies of the JSONL
splits from `run_sft.py`), `manifests/` (one JSON per `run_sft.py` training job), `logs/`
(local command output), `cdk/.venv/`, `cdk/cdk.out/`. `.ipynb_checkpoints/` folders are
JupyterLab autosave copies (they may hold outdated code; they aren't used).

### How the pieces fit

- **`pipeline/steps/*`** hold the logic as ordinary functions, so the same code runs from
  the scripts in Studio and inside the pipeline steps.
- **`pipeline/definition.py` and `deploy_definition.py`** wrap those functions as `@step`
  functions. Each `@step` runs as a small CPU job (`ml.m5.large`) in the SageMaker
  Distribution 4.6 image, which has the SDK. The project isn't installed in that image, so
  its code is shipped inside the pickled step functions
  (`cloudpickle.register_pickle_by_value`). This only works for modules imported **at
  module level**: an `import` inside a step function body fails in the step job.
- **`cdk/app.py`** calls `build_pipeline()` and `build_deploy_pipeline()` at synth time and
  embeds the JSON definitions in `AWS::SageMaker::Pipeline` resources. Building a
  definition uploads the serialized step code to the SageMaker default bucket
  (`s3://sagemaker-<region>-<account>/<pipeline name>/`). **Config values are baked into
  the definitions at synth time, so after editing `config/pipeline.yaml` you must run
  `./cdk.sh deploy`.** Only pipeline parameters can change per execution.
- **`ServerlessSFT` is a `@step`, not a native `TrainingStep`,** because
  `SFTTrainer.train()` calls `TrainingJob.create()` directly and isn't pipeline-aware. The
  step submits the serverless job and waits for it.
- **`ManualApproval` is a `CallbackStep`.** SageMaker posts a callback token to the SQS
  queue and waits until `scripts/approve_deployment.py` returns it with success or failure.
- **The endpoint uses inference components.** One *base* component serves Gemma 4 with vLLM
  (from the JumpStart hosting artifacts). Each deployment adds an *adapter* component with
  that version's LoRA weights (`<model artifacts>/checkpoints/hf/`). The smoke test runs
  against the new adapter before old adapters are removed.

## Prerequisites

- A **SageMaker Studio JupyterLab space** with SageMaker Distribution 4.x (Python 3.12, SDK
  v3, Node.js). Synth must run on **Python 3.12**, because the step image unpickles the step
  code.
- A Region with serverless model customization: `us-west-2` (used here), `us-east-1`,
  `ap-northeast-1` or `eu-west-1`.
- AWS credentials that can deploy CloudFormation/IAM (the Studio execution role here).
- The account **bootstrapped for CDK**: `cd cdk && ./cdk.sh bootstrap` (once per account
  and Region).
- An **MLflow app** in the domain; its ARN goes in `config/pipeline.local.yaml` (see
  below). Studio lists apps under **MLflow → Apps** (not "tracking servers"). To find it:
  `aws sagemaker list-mlflow-apps`.
- **Endpoint quota** for `deploy.instance_type`
  (Service Quotas → SageMaker → `ml.g7.2xlarge for endpoint usage` ≥ 1).

## First-time setup

```bash
git clone https://github.com/amitkalawat/sagemaker-serverless-fine-tuning-mlops.git
cd sagemaker-serverless-fine-tuning-mlops

# 1. Account-specific settings live in a gitignored local file
cp config/pipeline.local.example.yaml config/pipeline.local.yaml
#    then edit it: mlflow.tracking_server_arn, deploy.accept_eula (after reviewing the
#    model license), stack.notification_email (optional).
#    Review config/pipeline.yaml too: region, deploy.instance_type, stack.* names.

# 2. Deploy the stack (creates cdk/.venv on first run; synth builds both pipelines)
cd cdk
./cdk.sh synth
./cdk.sh diff
./cdk.sh deploy
cd ..

# 3. Confirm everything exists
python3 scripts/status.py
```

`./cdk.sh deploy` prints the stack outputs: the pipeline names, `ApprovalQueueUrl`,
`NotificationTopicArn`, `DataBucketName`, `EndpointName`. If you set
`stack.notification_email`, confirm the SNS subscription email. Approval requests and
pipeline results are then emailed to you.

## Runbook: run a training and deploy cycle

Run all commands from the repository root. A full cycle takes about an hour:

| Phase | Typical time |
|-------|--------------|
| Training pipeline (6 steps) | about 35 min: about 10 min of serverless training, plus 3-5 min of start-up for each `@step` job |
| Deploy pipeline up to `ManualApproval` | about 5 min (then it waits for you, indefinitely) |
| After approval, until the endpoint is serving | about 15 min on later deploys (the adapter loads in seconds); add about 7 min on the very first deploy (endpoint + base model) |

### Step 1: Decide what changes in this run

Edit `config/pipeline.yaml`. Common changes:

| Goal | Change |
|------|--------|
| More or different training data | `data.train_size`, `data.categories`, `data.max_chars`, `data.seed` |
| Stronger fine-tune | `hyperparameters.max_epochs`, `learning_rate`, `lora_rank` / `lora_alpha` (see [Getting a stronger fine-tune](#getting-a-stronger-fine-tune)) |
| Longer examples | `hyperparameters.dataset_max_len` (and `data.max_chars`) |
| Stricter or looser gate | `quality_gate.max_value`, or per run with the `QualityGateMaxValue` parameter (step 3) |
| Different endpoint size | `deploy.instance_type` **and** `deploy.compute` (see [Troubleshooting](#troubleshooting)) |

Re-running with an unchanged config is fine too. It retrains on the same splits (same
`data.seed`) and creates a new model package version.

### Step 2: Redeploy the stack if you changed the config

```bash
cd cdk && ./cdk.sh diff && ./cdk.sh deploy && cd ..
```

Skip this if you didn't edit `config/pipeline.yaml` or the code. The diff shows
`AWS::SageMaker::Pipeline` changes on every deploy, because each synth uploads freshly
serialized step code. That's expected.

### Step 3: Start training

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name gemma4-e4b-serverless-sft-cdk \
  --pipeline-execution-display-name "train-$(date +%Y%m%d-%H%M)"

# Optional: override the gate threshold for this run only
#   --pipeline-parameters Name=QualityGateMaxValue,Value=1.3
```

### Step 4: Monitor

```bash
python3 scripts/status.py            # both pipelines, model versions, endpoint
python3 scripts/status.py --last 3   # include earlier executions
```

Or use Studio: **Pipelines → gemma4-e4b-serverless-sft-cdk → Executions** shows the graph
and each step's logs. A failed step's reason appears in `status.py` output. Its full log
is in CloudWatch (`/aws/sagemaker/TrainingJobs`, stream `pipelines-<execution id>-<Step>-...`)
and attached to the MLflow summary run.

When training **succeeds**, the model package is `PendingManualApproval`, and EventBridge
starts the deploy pipeline automatically. You'll see a new deploy execution in
`status.py`. If the gate **fails**, the execution ends at `QualityGateFailed`, and no
deployment starts.

### Step 5: Review the candidate in MLflow

Open your MLflow app in Studio (**MLflow → Apps**), or run
`aws sagemaker create-presigned-mlflow-app-url --arn <mlflow.tracking_server_arn>`.
Experiment `gemma4-e4b-serverless-sft`, run `pipeline-<training execution id>`.
Check the [metrics for this use case](#metrics-for-this-sft-use-case): the `sft/train/loss`
curve, `sft/train/grad_norm` and `quality_gate/value`. Compare with earlier runs using the
MLflow **Compare** view.

### Step 6: Approve or reject

When `status.py` shows the deploy execution at `ManualApproval  Executing`:

```bash
QUEUE=$(aws cloudformation describe-stacks --stack-name Gemma4ServerlessSftPipeline \
  --query "Stacks[0].Outputs[?OutputKey=='ApprovalQueueUrl'].OutputValue" --output text)

python3 scripts/approve_deployment.py --queue-url $QUEUE --list      # what's waiting
python3 scripts/approve_deployment.py --queue-url $QUEUE --approve
#   or
python3 scripts/approve_deployment.py --queue-url $QUEUE --reject --reason "loss too high"
```

- **Approve** records your IAM identity as the approver, sets the package to `Approved`,
  and continues to `DeployEndpoint`.
- **Reject** fails the execution and sets the package to `Rejected`. The endpoint keeps
  serving the previous version.
- With several waiting executions, add `--execution-arn <arn>` (shown by `--list`).
- There's no timeout: an unanswered approval waits, and nothing is deployed.

### Step 7: Watch the deployment and verify the model

```bash
python3 scripts/status.py         # DeployEndpoint -> SmokeTest; endpoint components
python3 scripts/test_endpoint.py  # fine-tuned vs base on held-out eval examples
python3 scripts/test_endpoint.py --prompt "Summarize this: ..." --no-base
```

`SmokeTest` sends `deploy.smoke_test_examples` held-out prompts to the **new** adapter. Only
after they all succeed does it delete older adapters, so the endpoint never stops serving.
The MLflow run `deploy-<deploy execution id>` holds the results
(`smoke_test/results.json`).

To call the model from your own code, use the OpenAI-style chat format served by the vLLM
container, and target the adapter component:

```python
import boto3, json
rt = boto3.client("sagemaker-runtime")
resp = rt.invoke_endpoint(
    EndpointName="gemma4-e4b-sft-dolly",
    InferenceComponentName="gemma4-e4b-sft-dolly-<deploy execution id>",  # see status.py
    ContentType="application/json",
    Body=json.dumps({"messages": [{"role": "user", "content": "Summarize: ..."}],
                     "max_tokens": 256, "temperature": 0}),
)
print(json.loads(resp["Body"].read())["choices"][0]["message"]["content"])
```

Use the base component's name (`gemma4-e4b-sft-dolly-inference-component`) to query the
un-tuned model.

### Step 8: Roll back or redeploy a specific version

Start the deploy pipeline by hand with an explicit model package:

```bash
aws sagemaker start-pipeline-execution --pipeline-name gemma4-e4b-sft-deploy-cdk \
  --pipeline-parameters \
    Name=ModelPackageArn,Value=arn:aws:sagemaker:us-west-2:<account>:model-package/gemma4-e4b-sft-dolly-cdk/<version>
```

It goes through approval again. The default, `ModelPackageArn=latest`, takes the newest
`PendingManualApproval` package. `EndpointName` and `InstanceType` are parameters too. Only
packages registered by `RegisterCandidate` (version 2 onward) carry the metadata (eval data
URI, training job, MLflow run IDs) that the deploy pipeline uses.

### Step 9: Clean up when you're done

The endpoint is billed hourly while it exists. See [Cost and cleanup](#cost-and-cleanup).

## Getting a stronger fine-tune

The test cycle below used the defaults: 512 examples, 2 epochs, `global_batch_size` 128.
That's only **8 optimizer steps** (steps = `train_size × max_epochs / global_batch_size`).
The loss fell (2.23 → 1.18), but at temperature 0 the fine-tuned model answers almost
word-for-word like the base model. For example, it keeps the base model's long, formatted
answers (about 115 tokens) instead of Dolly's short references (about 30 words). So the
pipeline works end to end, but this configuration barely changes behavior. For a visible
effect, try:

| Setting | Default | Try | Why |
|---------|---------|-----|-----|
| `data.train_size` | 512 | 4096 (closed_qa + information_extraction + summarization have about 4,400 examples under 6,000 chars) | 32 steps per epoch instead of 4 |
| `hyperparameters.max_epochs` | 2 | 3 | 96 steps total with the line above |
| `hyperparameters.learning_rate` | 1e-4 | 2e-4 | a common LoRA learning rate; watch `sft/train/grad_norm` |
| `hyperparameters.lora_rank` / `lora_alpha` | 16 / 32 | 32 / 64 | more adapter capacity |
| `quality_gate.max_value` | 1.5 | retune after one run | the final loss shifts with data and steps |

Training time grows with the step count. In the 8-step test, the job spent about 6 min
downloading the image, 3.7 min in its Training phase (including model load) and about 1 min
uploading. Budget roughly 30-60 min of serverless training for about 100 steps until
you've measured it. Then compare runs on `sft/train/per_token_loss` in MLflow, and on
outputs with `scripts/test_endpoint.py`.

## What to look at in MLflow

Experiment **`gemma4-e4b-serverless-sft`** in the MLflow app holds three kinds of runs:

| Run | Written by | Use it for |
|-----|-----------|------------|
| `pipeline-<training execution id>` | every training `@step` | the overview of one training execution: params, data stats, the gate, ARNs, copied training curves, all logs |
| `sft-<training execution id>` | the serverless SFT job itself | raw training telemetry (about 30 metrics per step, about 140 resolved recipe params) |
| `deploy-<deploy execution id>` | every deploy `@step` | who approved what, which component was deployed, smoke-test latency and outputs |

Runs are linked by tags (`pipeline_execution_id`, `sft_mlflow_run_id`,
`training_pipeline_execution_id`, `deployed_endpoint`). The model package's
`CustomerMetadataProperties` hold the same IDs. Standalone `scripts/run_sft.py` runs appear
as `pipeline-local-<timestamp>` and `sft-local-<timestamp>`.

### Metrics for this SFT use case

The Gemma 4 SFT recipe logs **training metrics only**. It logs no validation loss, even
though a validation set is supplied. So judge a run on the training curves plus the
held-out outputs (smoke test, `test_endpoint.py`). In the summary run, the SFT metrics are
copied with an `sft/` prefix.

| Metric | What to look for |
|--------|------------------|
| **`sft/train/loss`** | The headline curve. It should fall steadily; the test runs went from 2.23 to about 1.18 over 8 steps. Its final value is the quality gate (`quality_gate/value`, threshold `QualityGateMaxValue`, default 1.5). Flat: the learning rate is too low or there are too few steps. Very low values (well under ~0.5) after several epochs suggest memorization, so check the held-out outputs. |
| **`sft/train/per_token_loss`** | Loss normalized per completion token, comparable across runs with different data mixes and lengths. Prefer it over `train/loss` when comparing runs. |
| `sft/train/perplexity` | exp(per-token loss): an interpretable view of the same signal. |
| **`sft/train/grad_norm`** | Training stability. Spikes or steady growth indicate instability: lower `learning_rate`, or add warmup (`warmup_steps`). |
| `sft/train/lr` | Confirms the schedule you configured (cosine with 5 warmup steps by default). |
| `sft/train/loss_mask_ratio`, `sft/train/num_valid_tokens` | How much of each sequence is trained on (the completion tokens). If valid tokens are tiny, prompts dominate or completions are being truncated. |
| `data/train_seq_len_max` (in the `sft-` run), `data/train/prompt_chars_max` | Check against `dataset_max_len` (2048 tokens). Longer examples get truncated, so raise `dataset_max_len` or filter with `data.max_chars`. |
| `data/<split>/examples`, `data/<split>/{prompt,completion}_chars_mean` | Did data prep produce what you expected (512/64/64 by default)? |
| `sft/perf/throughput`, `sft/perf/mfu/train`, `sft/perf/gpu_peak_memory_allocated_gb` | Efficiency and headroom. Spare GPU memory means you can raise `dataset_max_len` or the batch size. |
| `quality_gate/value`, `quality_gate/passed` | The gate decision recorded for the execution. |
| **`smoke_test/*`** (deploy run) | `latency_s_mean`/`latency_s_max` and `completion_tokens_mean` per request. Read **`smoke_test/results.json`**, which puts prompt, reference completion and model response side by side. |

Artifacts to check when something goes wrong: `logs/ServerlessSFT-job/*.log` (the training
container's VERL logs) and `logs/steps/<Step>/*.log` (each step job). Data samples are in
`data/*_sample.json`.

## Configuration

`config/pipeline.yaml` is the single source of configuration for the scripts, both
pipeline definitions and the CDK stack. `config/pipeline.local.yaml` (gitignored) is merged
over it. Keep account-specific values (the MLflow app ARN, the EULA acceptance, the email)
and any personal overrides there. **Redeploy the stack after editing either file.**

| Key | Meaning |
|-----|---------|
| `region` | AWS Region for everything |
| `base_model` | SageMaker Public Hub model ID |
| `model_package_group_name`, `pipeline_name` | names used by `run_sft.py` and the non-CDK `--upsert` path |
| `data.*` | Dolly source URL, categories, max length, split sizes, seed |
| `hyperparameters.*` | SFT recipe overrides. Allowed values: `global_batch_size` 128/256/512/1024, `lora_rank` 8/16/32/64/128, `lora_alpha` 16/32/64/128/256, `dataset_max_len` 256-32768, `learning_rate` 1e-7 to 1e-3, `max_epochs` 1-100. |
| `quality_gate.metric`, `.max_value` | the gate metric and default threshold (the `QualityGateMaxValue` pipeline parameter) |
| `mlflow.tracking_server_arn`, `.experiment_name` | MLflow app (set the ARN in the local file) and experiment. An empty ARN disables tracking. |
| `deploy.endpoint_name`, `.instance_type` | endpoint defaults (also deploy pipeline parameters) |
| `deploy.accept_eula` | must be `true` for ModelBuilder to use the base model's JumpStart hosting artifacts. It's `false` in the committed config: review the license (Gemma 4 E4B-IT is Apache-2.0) and set it in your local file. |
| `deploy.smoke_test_examples` | held-out examples the smoke test sends |
| `deploy.compute` | GPU/vCPU/memory reserved for the base component; must fit `instance_type` |
| `step_instance_type`, `step_image_uri` | compute and image for the `@step` jobs (the image is pinned by digest) |
| `stack.*` | names of the CDK-managed resources (kept apart from the names the scripts use), and `notification_email` |

## AWS resources

Created by the CDK stack `Gemma4ServerlessSftPipeline`:

| Resource | Purpose |
|----------|---------|
| S3 bucket `sagemaker-sft-pipeline-<account>-<region>` | training data and model artifacts (versioned, SSE-S3, TLS only, retained) |
| IAM role `service-role/SageMaker-gemma4-e4b-serverless-sft-cdk-<region>` | both pipelines, their step jobs, the SFT job and the endpoint model. It uses `AmazonSageMakerModelCustomizationCoreAccess` plus scoped extras (bucket access, dataset hub, passing itself, listing its own pipelines' steps, invoking/tagging endpoints, deleting its unused models, the queue, the topic). |
| Model package group `gemma4-e4b-sft-dolly-cdk` | fine-tuned model versions (retained) |
| Pipelines `gemma4-e4b-serverless-sft-cdk`, `gemma4-e4b-sft-deploy-cdk` | training and deploy |
| SQS queue `gemma4-e4b-sft-deploy-cdk-approvals` | ManualApproval callback tokens |
| SNS topic `gemma4-e4b-serverless-sft-cdk-notifications` | approval requests and both pipelines' final status |
| EventBridge rule `DeployOnTrainingSuccess` (+ role) | starts the deploy pipeline when training succeeds |
| EventBridge rule `PipelineExecutionStatus` | sends both pipelines' final status to SNS |

Not managed by CloudFormation: the **endpoint** `gemma4-e4b-sft-dolly`, its endpoint
config, inference components and base `Model` (all created by the deploy pipeline); the
serialized step code in the SageMaker default bucket; the MLflow app; and the registered
datasets in the SageMaker AI registry.

The role and bucket names contain `SageMaker`/`sagemaker`, and the role sits under
`service-role/`, because the managed policy scopes `iam:PassRole` and S3 access by those
patterns. The queue URL and topic ARN are deterministic, because they're baked into the
deploy pipeline definition at synth time.

## Iterating without CDK

For quick experiments from Studio, using the Studio execution role and the
non-`stack` names in the config:

```bash
python3 scripts/run_sft.py --dry-run      # data prep + registration + trainer, no training
python3 scripts/run_sft.py                # full training run in-process, tracked in MLflow
python3 scripts/run_sft.py --no-wait      # submit and exit
python3 pipeline/definition.py --output definition.json         # inspect the training definition
python3 pipeline/definition.py --upsert --start                  # create + run it outside CDK
python3 pipeline/deploy_definition.py --queue-url URL --topic-arn ARN --output deploy.json
```

## Cost and cleanup

- **Training run:** serverless GPU time for the SFT job (about 10 minutes with the
  defaults), plus six `ml.m5.large` step jobs of a few minutes each.
- **Deploy run:** five `ml.m5.large` step jobs; the approval wait itself costs nothing.
- **Endpoint:** `ml.g7.2xlarge`, billed per hour **while the endpoint exists**, whether or
  not it's used. The pipeline creates it, and CloudFormation doesn't manage it, so
  `cdk destroy` won't remove it. Delete it with:

  ```bash
  EP=gemma4-e4b-sft-dolly
  for ic in $(aws sagemaker list-inference-components --endpoint-name-equals $EP \
               --query 'InferenceComponents[].InferenceComponentName' --output text); do
    aws sagemaker delete-inference-component --inference-component-name $ic; done
  # wait until list-inference-components is empty (adapters go first; rerun if the base
  # component's deletion is rejected), then:
  aws sagemaker delete-endpoint --endpoint-name $EP
  aws sagemaker delete-endpoint-config --endpoint-config-name $EP
  aws sagemaker list-models --name-contains $EP-base --query 'Models[].ModelName' --output text \
    | xargs -n1 -r aws sagemaker delete-model --model-name
  ```

  The next approved deployment recreates the endpoint.
- **Everything else:** `cd cdk && ./cdk.sh destroy` removes the pipelines, queue, topic,
  rules and roles. It keeps the data bucket and the model package group (`RETAIN`), because
  model versions point at them; delete those by hand if you're done with the models.

## Troubleshooting

Problems hit while building this, and their fixes, which are already in the code:

| Symptom | Cause / fix |
|---------|-------------|
| Step fails: `ModuleNotFoundError: No module named 'pipeline'` | A step function imported project code inside its body. Import project modules at module level in `definition.py` / `deploy_definition.py` so cloudpickle ships them. |
| `iam:PassRole` denied on `arn:aws:iam::...:role/SageMaker-...` (no `service-role/` path) | An SDK call derived the role from the assumed-role identity, which drops the path. Pass the role ARN explicitly (as `register_dataset(..., role)` does); the stack also allows both ARN forms. |
| `ContainerArguments cannot contain empty string` | A `@step` argument was `""` (for example a pipeline parameter default). Use a sentinel such as `latest`. |
| CDK: `No module named 'aws_cdk'` | The CLI didn't start the app with the venv on `PATH`. `cdk.json` runs `.venv/bin/python app.py`; use `./cdk.sh`. |
| CDK: `Cannot use the jsii package cache ... /home ... (mode 0777)` | A Studio quirk; `cdk.sh` sets `JSII_RUNTIME_PACKAGE_CACHE=disabled`. |
| `CreateInferenceComponent`: `Number of accelerator devices required must be specified` | The SDK has no compute defaults for the instance type; set `deploy.compute.num_accelerators`. |
| Base component `Failed`: `not enough hardware resources` | `deploy.compute` asks for more host CPU/memory than the instance offers after SageMaker's reservation. On `ml.g7.2xlarge`, 4 vCPUs + 16 GB fails and 2 vCPUs + 8 GB works. Delete the failed component and redeploy. |
| Deploy `ResolveModel`: `No PendingManualApproval model package` | Nothing is waiting (for example the candidate was already approved or rejected). Pass `ModelPackageArn` explicitly. |
| MLflow summary run stuck `RUNNING` | A step crashed before it could log. The pipeline execution status is authoritative; set the run's status in MLflow if you want it tidy. |

## Test record

Verified end to end in `us-west-2` on 2026-10-27, running as the stack's role:

- **Training:** all 6 steps succeeded. The serverless SFT
  took about 10 min; `train/loss` went 2.23 → 1.19 (gate 1.5: pass); version 2 was
  registered as `PendingManualApproval`.
- **Event trigger:** the deploy pipeline started within 1 s of training succeeding.
- **Deploy:** it waited at `ManualApproval` until
  `approve_deployment.py --approve`; the package was set to `Approved` with the approver
  recorded; the adapter was deployed onto the existing endpoint (`InService` in about 7 s);
  the smoke test passed on 3 held-out prompts (mean 2.8 s, max 4.4 s); the old adapter was
  removed.
- **Endpoint:** fine-tuned and base components both answer correctly; an unknown
  component returns `ValidationError`; the outputs match the base model closely (see
  [Getting a stronger fine-tune](#getting-a-stronger-fine-tune)).
- **MLflow:** summary, SFT and deploy runs were all populated and cross-linked as described
  above.
- **Not yet exercised:** the reject path of `approve_deployment.py` in a live execution.

## Security

**What's in place**

- **No credentials in code or config.** Everything authenticates with IAM roles. The
  account-specific values (the MLflow app ARN, the EULA acceptance, the notification email)
  live in the gitignored `config/pipeline.local.yaml`. `.gitignore` also excludes runtime
  output (`logs/`, `manifests/`, `data/`, `cdk/cdk.out/`), which contains account IDs,
  ARNs and data copies.
- **Human approval gate.** Nothing is deployed without a callback from
  `approve_deployment.py`. The pipeline role can send to the approval queue, but it can't
  read the callback token or call `SendPipelineExecutionStepSuccess`, so it can't approve
  its own deployment. Approvals and rejections record the caller's IAM identity on the
  model package and in MLflow.
- **Storage.** The data bucket blocks public access, enforces TLS, uses SSE-S3 and is
  versioned. The approval queue uses SSE-SQS and enforces TLS.
- **Scoped extras.** On top of the managed policy, the role's extra permissions are scoped
  to this stack's buckets, queue, topic and pipelines, and to resources named after the
  configured endpoint. The EventBridge role can only start the deploy pipeline.
- **Pinned inputs.** The step image is pinned by digest, the CDK CLI and libraries by
  version, and the Dolly dataset by revision.

**Residual risks to address before production use**

| Area | Risk | Recommendation |
|------|------|----------------|
| IAM | The pipeline role uses the AWS managed `AmazonSageMakerModelCustomizationCoreAccess`. That policy is broad: it can create, update and delete endpoints; create and delete Lambda functions whose names contain `SageMaker`; run Bedrock import and evaluation; pass any `service-role/*SageMaker*` role; and read and write every bucket whose name contains `sagemaker`. | Replace it with a custom policy limited to the actions this project uses (see `cdk/stacks/sft_pipeline_stack.py`). |
| Step code integrity | Each `@step`'s pickled code and its SHA-256 (`metadata.json`) are stored side by side in the SageMaker default bucket under `<pipeline name>/`. Anyone who can write there can run code as the pipeline role. With the managed policy, that includes other roles that also have it, such as Studio execution roles. | Use a dedicated artifact bucket with a bucket policy that allows writes only from the deploying principal and the pipeline role. Evaluate the SDK's signed-payload mode. |
| Approval authority | Anyone with `sqs:ReceiveMessage` on the approval queue plus `sagemaker:SendPipelineExecutionStepSuccess` can approve. | For separation of duties, grant both only to an approvers role (IAM, plus a queue policy), and keep it separate from the people who change training. |
| Network | Step jobs, the SFT job and the endpoint run without a VPC. The `PrepareData` step downloads the dataset from the internet. | Mirror the dataset to S3. Run the jobs in private subnets (`SFTTrainer(networking=...)`, and `subnets`/`security_group_ids` on `@step`) with VPC endpoints. Consider network isolation for the endpoint model. |
| Data in MLflow | Data samples, smoke-test prompts and responses, and full CloudWatch job logs are copied into MLflow artifacts (S3). | With non-public data, treat the MLflow app and its artifact bucket as sensitive, or turn off the sample and log uploads in `pipeline/steps/tracking.py` call sites. |
| Encryption | No customer-managed KMS keys are used. The SNS topic is unencrypted, because EventBridge can't publish to a topic encrypted with the AWS-managed SNS key. | If you need it, use a CMK for the bucket, queue and topic, with key policies for SageMaker, EventBridge and SNS. |
| EULA | `deploy.accept_eula` defaults to `false`. Setting it accepts the base model's license on your behalf. | Review the license before setting it in your local config. |

Before pushing, check that nothing account-specific is staged:

```bash
git status --ignored             # local config and runtime folders should be listed as ignored
git grep -nE '[0-9]{12}' -- . ':!README.md' | grep -v 542918446943   # AWS's public ECR account is fine
```

## Known limitations

- **No held-out evaluation in the gate.** The recipe logs no validation loss, so the gate
  checks convergence (final `train/loss`), not quality. The next step is a serverless
  evaluation job on `eval.jsonl` (`LLMAsJudgeEvaluator` / `BenchMarkEvaluator`, see
  `docs/DESIGN.md`) feeding the gate.
- **Bedrock isn't available for Gemma 4.** Custom Model Import doesn't support the
  architecture. For a supported model (for example Qwen3 or Llama 3.x), import from the
  SFT output's `checkpoints/hf_merged/` (merged full weights).
- **`ml.g7.2xlarge` isn't on the hub's tested list** for this model (that list is the
  g6/g6e/p5 family). It works with the explicit `deploy.compute` above; retune
  `deploy.compute` if you change `deploy.instance_type`.
- **One candidate at a time.** The trigger passes no parameters, so `ResolveModel` takes
  the newest `PendingManualApproval` package in the group.
- **The endpoint is outside CloudFormation,** because it's created per deployment by the
  pipeline; see [Cost and cleanup](#cost-and-cleanup).

## License

MIT, see [LICENSE](LICENSE). The base model (Gemma 4 E4B-IT, Apache-2.0) and the
training data (Databricks Dolly 15k, CC BY-SA 3.0) have their own licenses. Neither is
included in this repository.
