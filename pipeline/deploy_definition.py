"""SageMaker Pipeline that deploys an SFT model package to a real-time endpoint, after approval.

    ResolveModel -> ManualApproval (CallbackStep, SQS) -> ApproveModelPackage
                 -> DeployEndpoint -> SmokeTest

Triggered by EventBridge when the training pipeline succeeds (see the CDK stack), or started
by hand. With `ModelPackageArn` left at its default, `latest`, it deploys the newest
PendingManualApproval package in the group, which the training pipeline's RegisterCandidate
step produced.

ManualApproval waits until someone runs scripts/approve_deployment.py, which reads the
callback token from the SQS queue and approves or rejects. ResolveModel publishes an SNS
message when approval is needed.

Usage (from Studio):
    python3 pipeline/deploy_definition.py --output deploy-definition.json
    python3 pipeline/deploy_definition.py --upsert --role ARN --queue-url URL --topic-arn ARN
"""

import argparse
import json
import sys
from pathlib import Path

import cloudpickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline as project_pkg  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.steps import deploy, tracking  # noqa: E402

from sagemaker.core.helper.session_helper import Session  # noqa: E402
from sagemaker.core.shapes import MlflowConfig  # noqa: E402
from sagemaker.core.workflow.execution_variables import ExecutionVariables  # noqa: E402
from sagemaker.core.workflow.parameters import ParameterString  # noqa: E402
from sagemaker.mlops.workflow.callback_step import (  # noqa: E402
    CallbackOutput,
    CallbackOutputTypeEnum,
    CallbackStep,
)
from sagemaker.mlops.workflow.function_step import step  # noqa: E402
from sagemaker.mlops.workflow.pipeline import Pipeline  # noqa: E402

# Same code-shipping rule as pipeline/definition.py: modules used by step functions must be
# imported at module level so cloudpickle ships them by value.
cloudpickle.register_pickle_by_value(project_pkg)


def _deploy_run(mlflow_cfg: dict, execution_id: str, **tags):
    arn = mlflow_cfg.get("tracking_server_arn")
    if not arn:
        return None, None
    run_id = tracking.get_or_create_run(
        arn, mlflow_cfg["experiment_name"], execution_id, tags, run_name_prefix="deploy"
    )
    return arn, run_id


def resolve_model(
    model_package_arn: str,
    model_package_group_name: str,
    topic_arn: str,
    approval_queue_url: str,
    mlflow_cfg: dict,
    execution_id: str,
    execution_arn: str,
) -> dict:
    import boto3

    model = deploy.resolve_model_package(model_package_arn, model_package_group_name)
    arn, run_id = _deploy_run(
        mlflow_cfg, execution_id,
        pipeline_kind="deploy",
        model_package_arn=model["model_package_arn"],
        training_job_name=model["training_job_name"],
        training_pipeline_execution_id=model["metadata"].get("training_pipeline_execution_id", ""),
    )
    if run_id:
        tracking.log(arn, run_id, params={"model": model})
    message = (
        f"Deployment approval needed for {model['model_package_arn']} "
        f"(version {model['model_package_version']}, base {model['base_model']}).\n"
        f"Quality gate value: {model['metadata'].get('gate_value', 'n/a')}\n"
        f"Training job: {model['training_job_name']}\n"
        f"Deploy pipeline execution: {execution_arn}\n\n"
        "Review the training run in MLflow, then from the project directory run:\n"
        f"  python3 scripts/approve_deployment.py --queue-url {approval_queue_url} --approve\n"
        f"  python3 scripts/approve_deployment.py --queue-url {approval_queue_url} --reject "
        "--reason '...'\n"
    )
    boto3.client("sns").publish(
        TopicArn=topic_arn, Subject="SFT model deployment approval needed", Message=message
    )
    return model


def approve_model_package(model: dict, approver: str, mlflow_cfg: dict, execution_id: str) -> str:
    deploy.set_approval(model["model_package_arn"], "Approved", f"Approved for deployment by {approver}")
    arn, run_id = _deploy_run(mlflow_cfg, execution_id)
    if run_id:
        tracking.log(arn, run_id, tags={"approved_by": approver, "model_approval_status": "Approved"})
    return model["model_package_arn"]


def deploy_endpoint(
    model: dict,
    approved_arn: str,
    endpoint_name: str,
    instance_type: str,
    role: str,
    accept_eula: bool,
    compute: dict,
    mlflow_cfg: dict,
    execution_id: str,
) -> dict:
    component = f"{endpoint_name}-{execution_id}"[:63]
    arn, run_id = _deploy_run(mlflow_cfg, execution_id)
    if run_id:
        tracking.log(arn, run_id, params={"endpoint_name": endpoint_name,
                                          "instance_type": instance_type, "compute": compute})
    try:
        deployment = deploy.deploy_adapter(
            model["training_job_name"], endpoint_name, component, instance_type, role,
            accept_eula, compute,
        )
    except Exception as e:
        if run_id:
            tracking.log(arn, run_id, tags={"failure_reason": str(e)[:5000]})
            tracking.end_run(arn, run_id, "FAILED")
        raise
    if run_id:
        tracking.log(arn, run_id, tags={"inference_component_name": component})
    return deployment


def smoke_test(
    model: dict, deployment: dict, n_examples: int, mlflow_cfg: dict, execution_id: str,
    execution_arn: str,
) -> dict:
    examples = deploy.read_jsonl(model["metadata"]["eval_data_uri"], n_examples)
    arn, run_id = _deploy_run(mlflow_cfg, execution_id)
    try:
        report = deploy.smoke_test(
            deployment["endpoint_name"], deployment["inference_component_name"], examples
        )
    except Exception as e:
        if run_id:
            tracking.log(arn, run_id, tags={"failure_reason": str(e)[:5000],
                                            "smoke_test": "failed"})
            tracking.end_run(arn, run_id, "FAILED")
        raise
    removed = deploy.remove_other_adapters(
        deployment["endpoint_name"], keep=deployment["inference_component_name"]
    )
    if run_id:
        tracking.log(
            arn, run_id,
            metrics=report["metrics"],
            tags={"smoke_test": "passed", "removed_adapters": ",".join(removed) or "none"},
            json_artifacts={"smoke_test/results.json": report["results"]},
        )
        for step_name, job_name in tracking.pipeline_step_jobs(execution_arn).items():
            tracking.log_training_job_logs(arn, run_id, job_name, f"steps/{step_name}")
        # Point the training summary run at its deployment too.
        summary_run = model["metadata"].get("summary_mlflow_run_id")
        if summary_run:
            tracking.log(arn, summary_run, tags={
                "deployed_endpoint": deployment["endpoint_name"],
                "deployed_inference_component": deployment["inference_component_name"],
                "deploy_pipeline_execution_id": execution_id,
            })
        tracking.end_run(arn, run_id)
    return {**deployment, "removed_adapters": removed, **report["metrics"]}


def build_deploy_pipeline(
    cfg: dict,
    role_arn: str,
    artifact_bucket: str,
    approval_queue_url: str,
    topic_arn: str,
    model_package_group_name: str = None,
    pipeline_name: str = None,
) -> Pipeline:
    dcfg = cfg["deploy"]
    pipeline_name = pipeline_name or dcfg["pipeline_name"]
    model_package_group_name = model_package_group_name or cfg["model_package_group_name"]
    mlflow_cfg = cfg.get("mlflow") or {}
    session = Session(default_bucket=artifact_bucket)
    execution_id = ExecutionVariables.PIPELINE_EXECUTION_ID
    execution_arn = ExecutionVariables.PIPELINE_EXECUTION_ARN

    # "latest" (not ""): @step arguments become container arguments, which can't be empty.
    model_package_arn = ParameterString(name="ModelPackageArn", default_value=deploy.LATEST)
    endpoint_name = ParameterString(name="EndpointName", default_value=dcfg["endpoint_name"])
    instance_type = ParameterString(name="InstanceType", default_value=dcfg["instance_type"])

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

    model = as_step(resolve_model, "ResolveModel")(
        model_package_arn, model_package_group_name, topic_arn, approval_queue_url,
        mlflow_cfg, execution_id, execution_arn,
    )
    approval = CallbackStep(
        name="ManualApproval",
        description="Waits for scripts/approve_deployment.py --approve/--reject",
        sqs_queue_url=approval_queue_url,
        inputs={
            "pipeline_name": pipeline_name,
            "model_package_arn": model["model_package_arn"],
            "training_job_name": model["training_job_name"],
        },
        outputs=[CallbackOutput(output_name="approver", output_type=CallbackOutputTypeEnum.String)],
        depends_on=[model],
    )
    approved_arn = as_step(approve_model_package, "ApproveModelPackage")(
        model, approval.properties.Outputs["approver"], mlflow_cfg, execution_id
    )
    deployment = as_step(deploy_endpoint, "DeployEndpoint", max_runtime_in_seconds=3 * 60 * 60)(
        model, approved_arn, endpoint_name, instance_type, role_arn, bool(dcfg["accept_eula"]),
        dcfg["compute"], mlflow_cfg, execution_id,
    )
    result = as_step(smoke_test, "SmokeTest")(
        model, deployment, int(dcfg["smoke_test_examples"]), mlflow_cfg, execution_id,
        execution_arn,
    )
    return Pipeline(
        name=pipeline_name,
        parameters=[model_package_arn, endpoint_name, instance_type],
        # ManualApproval is only referenced through its output properties, which the step
        # compiler doesn't follow, so it has to be listed explicitly.
        steps=[approval, result],
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
    parser = argparse.ArgumentParser(description="Build, upsert or start the deploy pipeline")
    parser.add_argument("--role", help="pipeline/step execution role ARN (default: Studio role)")
    parser.add_argument("--queue-url", required=True, help="SQS queue URL for ManualApproval")
    parser.add_argument("--topic-arn", required=True, help="SNS topic for approval requests")
    parser.add_argument("--output", help="write the definition JSON to this file")
    parser.add_argument("--upsert", action="store_true", help="create or update the pipeline")
    args = parser.parse_args()

    from sagemaker.core.helper.session_helper import get_execution_role

    cfg = load_config()
    role = args.role or get_execution_role()
    pipeline = build_deploy_pipeline(
        cfg, role, Session().default_bucket(), args.queue_url, args.topic_arn
    )
    if args.output:
        Path(args.output).write_text(json.dumps(json.loads(pipeline.definition()), indent=2))
        print(f"Wrote {args.output}")
    if args.upsert:
        print(json.dumps(pipeline.upsert(role_arn=role), indent=2, default=str))


if __name__ == "__main__":
    main()
