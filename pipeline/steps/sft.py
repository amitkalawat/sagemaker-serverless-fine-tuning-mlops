"""Register datasets, run serverless SFT, and read back training metrics.

These functions are used both by scripts/run_sft.py (directly, from Studio) and by the
@step functions in pipeline/definition.py (inside the pipeline's step jobs).
"""

from botocore.exceptions import ClientError
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique
from sagemaker.core.resources import ModelPackageGroup, TrainingJob
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer


def ensure_model_package_group(name: str) -> ModelPackageGroup:
    try:
        return ModelPackageGroup.create(
            model_package_group_name=name,
            model_package_group_description="Serverless SFT (LoRA) fine-tunes of Gemma 4",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceInUse", "ValidationException"):
            return ModelPackageGroup.get(model_package_group_name=name)
        raise


def register_dataset(name: str, s3_uri: str, role: str = None) -> str:
    """Register a JSONL file in the SageMaker AI registry and return the dataset ARN.

    Pass `role` explicitly in step jobs: the SDK's default role is derived from the
    assumed-role identity, which drops the role's path (e.g. /service-role/), and that ARN
    doesn't match path-scoped iam:PassRole policies.
    """
    dataset = DataSet.create(
        name=name,
        source=s3_uri,
        customization_technique=CustomizationTechnique.SFT,
        wait=True,
        role=role,
    )
    return dataset.arn


def build_trainer(
    base_model: str,
    model_package_group: str,
    training_dataset: str,
    validation_dataset: str,
    s3_output_path: str,
    hyperparameters: dict,
    role: str = None,
    base_job_name: str = None,
    mlflow_resource_arn: str = None,
    mlflow_experiment_name: str = None,
    mlflow_run_name: str = None,
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=base_model,
        training_type=TrainingType.LORA,
        model_package_group=model_package_group,
        mlflow_resource_arn=mlflow_resource_arn,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        s3_output_path=s3_output_path,
        role=role,
    )
    if base_job_name:
        trainer.base_job_name = base_job_name
    for key, value in hyperparameters.items():
        setattr(trainer.hyperparameters, key, value)
    return trainer


def describe_result(training_job: TrainingJob) -> dict:
    """Summarize a finished serverless SFT job as plain JSON-serializable values."""
    training_job.refresh()
    mlflow_details = getattr(training_job, "mlflow_details", None)
    return {
        "training_job_name": training_job.training_job_name,
        "status": training_job.training_job_status,
        "failure_reason": getattr(training_job, "failure_reason", None),
        "model_package_arn": getattr(training_job, "output_model_package_arn", None),
        "mlflow_run_id": getattr(mlflow_details, "mlflow_run_id", None),
        "mlflow_resource_arn": getattr(
            getattr(training_job, "mlflow_config", None), "mlflow_resource_arn", None
        ),
    }


def data_stats(splits: dict) -> dict:
    """Example counts and character-length stats per split, as flat MLflow metrics."""
    stats = {}
    for split, rows in splits.items():
        stats[f"data/{split}/examples"] = len(rows)
        for field in ("prompt", "completion"):
            lengths = sorted(len(r[field]) for r in rows)
            stats[f"data/{split}/{field}_chars_mean"] = sum(lengths) / len(lengths)
            stats[f"data/{split}/{field}_chars_max"] = lengths[-1]
    return stats


def final_metric(mlflow_resource_arn: str, run_id: str, metric: str) -> float:
    """Return the last logged value of `metric` for an MLflow run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_resource_arn)
    history = MlflowClient().get_metric_history(run_id, metric)
    if not history:
        raise ValueError(f"Metric {metric!r} not found in MLflow run {run_id}")
    return max(history, key=lambda m: m.step).value
