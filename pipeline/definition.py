"""SageMaker Pipeline for serverless SFT of Gemma 4 E4B-IT.

    PrepareData -> RegisterDatasets -> ServerlessSFT -> Evaluate -> QualityGate
                                                                     |-- pass: RegisterCandidate
                                                                     `-- fail: FailStep

RegisterCandidate sets the model package to PendingManualApproval. Approval and deployment
happen in the separate deploy pipeline (pipeline/deploy_definition.py).

Each step is a @step function: a small CPU job (config `step_instance_type`) that calls the
SDK. `ServerlessSFT` submits the serverless training job and waits for it, because
`SFTTrainer.train()` calls `TrainingJob.create()` directly and can't be used as `step_args`
for a native `TrainingStep`.

Every step also logs to MLflow (config `mlflow`); see pipeline/steps/tracking.py.

Building the definition uploads the serialized step functions to `artifact_bucket`, so that
bucket must already exist (the CDK app uses the SageMaker default bucket).

Usage (from Studio):
    python3 pipeline/definition.py --output definition.json   # write the definition JSON
    python3 pipeline/definition.py --upsert --role ARN     # create/update the pipeline
    python3 pipeline/definition.py --upsert --start --role ARN
"""

import argparse
import json
import sys
from pathlib import Path

import cloudpickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline as project_pkg  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.steps import data_prep, sft, tracking  # noqa: E402

from sagemaker.core.helper.session_helper import Session  # noqa: E402
from sagemaker.core.shapes import MlflowConfig  # noqa: E402
from sagemaker.core.workflow.conditions import ConditionLessThanOrEqualTo  # noqa: E402
from sagemaker.core.workflow.execution_variables import ExecutionVariables  # noqa: E402
from sagemaker.core.workflow.parameters import ParameterFloat  # noqa: E402
from sagemaker.mlops.workflow.condition_step import ConditionStep  # noqa: E402
from sagemaker.mlops.workflow.fail_step import FailStep  # noqa: E402
from sagemaker.mlops.workflow.function_step import step  # noqa: E402
from sagemaker.mlops.workflow.pipeline import Pipeline  # noqa: E402

# The step jobs don't have this project installed, so ship its code inside the pickled
# step functions instead of pickling references to the `pipeline` package. This only covers
# modules the step functions reference as globals, so import them at module level above,
# not inside the function bodies (those imports would run, and fail, in the step job).
cloudpickle.register_pickle_by_value(project_pkg)


def _summary_run(mlflow_cfg: dict, execution_id: str, **tags):
    """(tracking ARN, summary run ID) for this execution, or (None, None) if tracking is off."""
    arn = mlflow_cfg.get("tracking_server_arn")
    if not arn:
        return None, None
    return arn, tracking.get_or_create_run(arn, mlflow_cfg["experiment_name"], execution_id, tags)


def prepare_data(
    data_cfg: dict, data_bucket: str, prefix: str, mlflow_cfg: dict, run_tags: dict,
    execution_id: str,
) -> dict:
    splits = data_prep.build_splits(data_cfg)
    uris = data_prep.write_and_upload(
        splits, "/tmp/data", f"s3://{data_bucket}/{prefix}/{execution_id}/data"
    )
    arn, run_id = _summary_run(mlflow_cfg, execution_id, **run_tags)
    if run_id:
        tracking.log(
            arn, run_id,
            params={"data": data_cfg},
            metrics=sft.data_stats(splits),
            tags={f"data_uri.{k}": v for k, v in uris.items()},
            json_artifacts={f"data/{k}_sample.json": v[:5] for k, v in splits.items()},
        )
    return uris


def register_datasets(
    uris: dict, model_package_group_name: str, role: str, mlflow_cfg: dict, execution_id: str
) -> dict:
    sft.ensure_model_package_group(model_package_group_name)
    datasets = {
        "train": sft.register_dataset(f"{model_package_group_name}-train", uris["train"], role),
        "validation": sft.register_dataset(
            f"{model_package_group_name}-val", uris["validation"], role
        ),
    }
    arn, run_id = _summary_run(mlflow_cfg, execution_id)
    if run_id:
        tracking.log(
            arn, run_id,
            tags={"model_package_group": model_package_group_name,
                  **{f"dataset_arn.{k}": v for k, v in datasets.items()}},
        )
    return datasets


def serverless_sft(
    datasets: dict,
    base_model: str,
    model_package_group_name: str,
    hyperparameters: dict,
    output_path: str,
    role: str,
    mlflow_cfg: dict,
    execution_id: str,
) -> dict:
    arn, run_id = _summary_run(mlflow_cfg, execution_id)
    trainer = sft.build_trainer(
        base_model=base_model,
        model_package_group=model_package_group_name,
        training_dataset=datasets["train"],
        validation_dataset=datasets["validation"],
        s3_output_path=output_path,
        hyperparameters=hyperparameters,
        role=role,
        mlflow_resource_arn=arn,
        mlflow_experiment_name=mlflow_cfg.get("experiment_name"),
        mlflow_run_name=f"sft-{execution_id}",
    )
    if run_id:
        tracking.log(
            arn, run_id,
            params={"base_model": base_model,
                    "hyperparameters": trainer.hyperparameters.to_dict()},
        )
    result = sft.describe_result(trainer.train(wait=True))
    if run_id:
        tracking.log(
            arn, run_id,
            tags={"training_job_name": result["training_job_name"],
                  "training_job_status": result["status"],
                  "model_package_arn": result["model_package_arn"],
                  "sft_mlflow_run_id": result["mlflow_run_id"]},
        )
        tracking.log_training_job_logs(arn, run_id, result["training_job_name"], "ServerlessSFT-job")
        if result["mlflow_run_id"]:
            tracking.client(arn).set_tag(result["mlflow_run_id"], tracking.EXECUTION_TAG, execution_id)
            tracking.copy_metric_history(arn, result["mlflow_run_id"], run_id, ("train/", "perf/"))
    if result["status"] != "Completed":
        if run_id:
            tracking.log(arn, run_id, tags={"failure_reason": result["failure_reason"]})
            tracking.end_run(arn, run_id, "FAILED")
        raise RuntimeError(f"Serverless SFT job did not complete: {result}")
    return result


def evaluate(
    training: dict, metric: str, max_value: float, mlflow_cfg: dict, execution_id: str,
    execution_arn: str,
) -> float:
    max_value = float(max_value)  # pipeline parameters reach @step functions as strings
    value = sft.final_metric(training["mlflow_resource_arn"], training["mlflow_run_id"], metric)
    arn, run_id = _summary_run(mlflow_cfg, execution_id)
    if run_id:
        passed = value <= max_value
        tracking.log(
            arn, run_id,
            params={"quality_gate.metric": metric, "quality_gate.max_value": max_value},
            metrics={"quality_gate/value": value, "quality_gate/passed": int(passed)},
            tags={"quality_gate": "passed" if passed else "failed"},
        )
        # Logs of the step jobs that have finished so far (this step's own log is attached
        # by RegisterCandidate).
        for step_name, job_name in tracking.pipeline_step_jobs(execution_arn).items():
            tracking.log_training_job_logs(arn, run_id, job_name, f"steps/{step_name}")
        if not passed:
            tracking.end_run(arn, run_id, "FAILED")
    return value


def register_candidate(
    training: dict, gate_value: float, data_uris: dict, mlflow_cfg: dict, execution_id: str,
    execution_arn: str,
) -> str:
    """Mark the gated model as a deployment candidate (PendingManualApproval).

    Human approval happens in the deploy pipeline, which the training pipeline's success
    event triggers. The metadata lets the deploy pipeline find the training job, the eval
    data and the MLflow runs without re-deriving them.
    """
    import boto3

    arn, run_id = _summary_run(mlflow_cfg, execution_id)
    metadata = {
        "training_job_name": training["training_job_name"],
        "training_pipeline_execution_id": execution_id,
        "gate_value": f"{float(gate_value):.6f}",
        "eval_data_uri": data_uris["eval"],
        "sft_mlflow_run_id": training["mlflow_run_id"] or "",
        "summary_mlflow_run_id": run_id or "",
    }
    boto3.client("sagemaker").update_model_package(
        ModelPackageArn=training["model_package_arn"],
        ModelApprovalStatus="PendingManualApproval",
        ApprovalDescription=f"Passed quality gate in training execution {execution_id}",
        CustomerMetadataProperties=metadata,
    )
    if run_id:
        evaluate_job = tracking.pipeline_step_jobs(execution_arn).get("Evaluate")
        if evaluate_job:
            tracking.log_training_job_logs(arn, run_id, evaluate_job, "steps/Evaluate")
        tracking.log(arn, run_id, tags={"model_approval_status": "PendingManualApproval"})
        tracking.end_run(arn, run_id)
    return training["model_package_arn"]


def build_pipeline(
    cfg: dict,
    role_arn: str,
    data_bucket: str,
    artifact_bucket: str,
    model_package_group_name: str = None,
    pipeline_name: str = None,
) -> Pipeline:
    pipeline_name = pipeline_name or cfg["pipeline_name"]
    model_package_group_name = model_package_group_name or cfg["model_package_group_name"]
    mlflow_cfg = cfg.get("mlflow") or {}
    session = Session(default_bucket=artifact_bucket)
    max_metric = ParameterFloat(
        name="QualityGateMaxValue", default_value=float(cfg["quality_gate"]["max_value"])
    )
    execution_id = ExecutionVariables.PIPELINE_EXECUTION_ID
    execution_arn = ExecutionVariables.PIPELINE_EXECUTION_ARN

    def as_step(func, name, **overrides):
        return step(
            func,
            name=name,
            role=role_arn,
            image_uri=cfg["step_image_uri"],
            instance_type=cfg["step_instance_type"],
            keep_alive_period_in_seconds=0,
            **overrides,
        )

    uris = as_step(prepare_data, "PrepareData")(
        cfg["data"], data_bucket, pipeline_name, mlflow_cfg,
        {"pipeline_name": pipeline_name, "base_model": cfg["base_model"]}, execution_id,
    )
    datasets = as_step(register_datasets, "RegisterDatasets")(
        uris, model_package_group_name, role_arn, mlflow_cfg, execution_id
    )
    training = as_step(serverless_sft, "ServerlessSFT", max_runtime_in_seconds=12 * 60 * 60)(
        datasets,
        cfg["base_model"],
        model_package_group_name,
        cfg["hyperparameters"],
        f"s3://{data_bucket}/{pipeline_name}/output/",
        role_arn,
        mlflow_cfg,
        execution_id,
    )
    gate_value = as_step(evaluate, "Evaluate")(
        training, cfg["quality_gate"]["metric"], max_metric, mlflow_cfg, execution_id,
        execution_arn,
    )

    gate = ConditionStep(
        name="QualityGate",
        conditions=[ConditionLessThanOrEqualTo(left=gate_value, right=max_metric)],
        if_steps=[
            as_step(register_candidate, "RegisterCandidate")(
                training, gate_value, uris, mlflow_cfg, execution_id, execution_arn
            )
        ],
        else_steps=[
            FailStep(
                name="QualityGateFailed",
                error_message=f"{cfg['quality_gate']['metric']} is above QualityGateMaxValue",
            )
        ],
    )
    return Pipeline(
        name=pipeline_name,
        parameters=[max_metric],
        steps=[gate],
        sagemaker_session=session,
        mlflow_config=(
            MlflowConfig(
                mlflow_resource_arn=mlflow_cfg["tracking_server_arn"],
                mlflow_experiment_name=mlflow_cfg["experiment_name"],
            )
            if mlflow_cfg.get("tracking_server_arn")
            else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Build, upsert or start the SFT pipeline")
    parser.add_argument("--role", help="pipeline/step execution role ARN (default: Studio role)")
    parser.add_argument("--data-bucket", help="bucket for data and outputs (default: SM bucket)")
    parser.add_argument("--output", help="write the definition JSON to this file")
    parser.add_argument("--upsert", action="store_true", help="create or update the pipeline")
    parser.add_argument("--start", action="store_true", help="start an execution after upsert")
    args = parser.parse_args()

    from sagemaker.core.helper.session_helper import get_execution_role

    cfg = load_config()
    artifact_bucket = Session().default_bucket()
    role = args.role or get_execution_role()
    pipeline = build_pipeline(cfg, role, args.data_bucket or artifact_bucket, artifact_bucket)

    if args.output:
        Path(args.output).write_text(json.dumps(json.loads(pipeline.definition()), indent=2))
        print(f"Wrote {args.output}")
    if args.upsert:
        print(json.dumps(pipeline.upsert(role_arn=role), indent=2, default=str))
    if args.start:
        execution = pipeline.start()
        print(f"Started execution: {execution.arn}")


if __name__ == "__main__":
    main()
