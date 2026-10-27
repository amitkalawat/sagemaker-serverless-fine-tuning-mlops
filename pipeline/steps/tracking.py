"""MLflow tracking for pipeline executions (training, deploy) and standalone runs.

Each execution gets one summary run in the configured experiment, found by its
`pipeline_execution_id` tag, which every step appends to. The serverless SFT job logs to its
own run in the same experiment. The summary run links to it, copies its training curves, and
stores the CloudWatch logs of the SFT job and the step jobs as artifacts.
"""

import json
import tempfile
from pathlib import Path

import boto3
import mlflow
from mlflow.tracking import MlflowClient

EXECUTION_TAG = "pipeline_execution_id"


def client(tracking_arn: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_arn)
    return MlflowClient()


def get_or_create_run(
    tracking_arn: str, experiment_name: str, execution_id: str, tags: dict = None,
    run_name_prefix: str = "pipeline",
) -> str:
    """Return the summary run ID for `execution_id`, creating `<prefix>-<id>` on first use."""
    c = client(tracking_arn)
    experiment = c.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else c.create_experiment(experiment_name)
    )
    # Match the name too: the SFT job's own run carries the same execution tag (for linking).
    run_name = f"{run_name_prefix}-{execution_id}"
    runs = c.search_runs(
        [experiment_id],
        filter_string=f"tags.{EXECUTION_TAG} = '{execution_id}' and attributes.run_name = '{run_name}'",
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id
    run = c.create_run(
        experiment_id,
        run_name=run_name,
        tags={EXECUTION_TAG: execution_id, **(tags or {})},
    )
    return run.info.run_id


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value if not isinstance(value, list) else ",".join(map(str, value))
    return out


def log(
    tracking_arn: str,
    run_id: str,
    params: dict = None,
    metrics: dict = None,
    tags: dict = None,
    json_artifacts: dict = None,
    text_artifacts: dict = None,
):
    """Log nested params, scalar metrics, tags, and {path: obj|text} artifacts to a run."""
    c = client(tracking_arn)
    for key, value in _flatten(params or {}).items():
        c.log_param(run_id, key, str(value)[:6000])
    for key, value in (metrics or {}).items():
        c.log_metric(run_id, key, float(value))
    for key, value in (tags or {}).items():
        c.set_tag(run_id, key, str(value))
    artifacts = {p: json.dumps(o, indent=2, default=str) for p, o in (json_artifacts or {}).items()}
    artifacts.update(text_artifacts or {})
    if artifacts:
        with tempfile.TemporaryDirectory() as tmp:
            for rel_path, text in artifacts.items():
                path = Path(tmp) / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            for path in Path(tmp).iterdir():
                if path.is_dir():
                    c.log_artifacts(run_id, str(path), artifact_path=path.name)
                else:
                    c.log_artifact(run_id, str(path))


def copy_metric_history(tracking_arn: str, src_run_id: str, dst_run_id: str, prefixes) -> int:
    """Copy the full step history of metrics starting with any of `prefixes`. Returns count."""
    from mlflow.entities import Metric

    c = client(tracking_arn)
    names = [k for k in c.get_run(src_run_id).data.metrics if k.startswith(tuple(prefixes))]
    for name in names:
        history = c.get_metric_history(src_run_id, name)
        for i in range(0, len(history), 1000):  # log_batch accepts at most 1000 metrics
            c.log_batch(
                dst_run_id,
                metrics=[Metric(f"sft/{m.key}", m.value, m.timestamp, m.step) for m in history[i : i + 1000]],
            )
    return len(names)


def training_job_logs(job_name: str, max_events: int = 20000) -> dict:
    """Return {stream_name: text} for a training job's CloudWatch log streams."""
    logs = boto3.client("logs")
    group = "/aws/sagemaker/TrainingJobs"
    streams = logs.describe_log_streams(logGroupName=group, logStreamNamePrefix=job_name)[
        "logStreams"
    ]
    out = {}
    for stream in streams:
        lines, token = [], None
        while len(lines) < max_events:
            kwargs = {"logGroupName": group, "logStreamName": stream["logStreamName"],
                      "startFromHead": True}
            if token:
                kwargs["nextToken"] = token
            resp = logs.get_log_events(**kwargs)
            lines += [e["message"] for e in resp["events"]]
            if resp["nextForwardToken"] == token:
                break
            token = resp["nextForwardToken"]
        out[stream["logStreamName"]] = "\n".join(lines[:max_events]) + "\n"
    return out


def log_training_job_logs(tracking_arn: str, run_id: str, job_name: str, folder: str):
    """Attach a training job's CloudWatch logs to the run under logs/<folder>/."""
    streams = training_job_logs(job_name)
    log(
        tracking_arn,
        run_id,
        text_artifacts={
            f"logs/{folder}/{name.split('/', 1)[-1].replace('/', '_')}.log": text
            for name, text in streams.items()
        },
    )
    return len(streams)


def pipeline_step_jobs(execution_arn: str) -> dict:
    """Return {step_name: training_job_name} for the finished @step jobs of an execution."""
    sm = boto3.client("sagemaker")
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)[
        "PipelineExecutionSteps"
    ]
    return {
        s["StepName"]: s["Metadata"]["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
        for s in steps
        if s.get("StepStatus") in ("Succeeded", "Failed") and "TrainingJob" in s.get("Metadata", {})
    }


def end_run(tracking_arn: str, run_id: str, status: str = "FINISHED"):
    client(tracking_arn).set_terminated(run_id, status=status)
