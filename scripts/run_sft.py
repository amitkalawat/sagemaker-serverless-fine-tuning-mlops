"""End-to-end test of serverless SFT (LoRA) on Gemma 4 E4B-IT, run directly from Studio.

Runs the same steps the pipeline runs, in-process:
  1. data prep      -> train/validation/eval JSONL in S3
  2. registration   -> model package group + SageMaker AI registry datasets
  3. serverless SFT -> fine-tuned model package
  4. quality gate   -> final value of the gate metric (MLflow) vs threshold

Everything is tracked in MLflow like a pipeline execution: summary run `pipeline-local-<ts>`
plus the SFT job's run `sft-local-<ts>` (config `mlflow`).

Usage:
  python3 scripts/run_sft.py --dry-run   # steps 1-2 + build the trainer, don't start training
  python3 scripts/run_sft.py             # full run, waits for training to finish
  python3 scripts/run_sft.py --no-wait   # start training and exit
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sagemaker.core.helper.session_helper import Session, get_execution_role  # noqa: E402

from pipeline.config import PROJECT_ROOT, load_config  # noqa: E402
from pipeline.steps import data_prep, sft, tracking  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="don't start the training job")
    parser.add_argument("--no-wait", action="store_true", help="start training and exit")
    args = parser.parse_args()

    cfg = load_config()
    session = Session()
    bucket = session.default_bucket()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    s3_prefix = f"s3://{bucket}/{cfg['pipeline_name']}/{run_id}"
    role = get_execution_role()

    print(f"[1/4] Preparing data -> {s3_prefix}/data/")
    splits = data_prep.build_splits(cfg["data"])
    uris = data_prep.write_and_upload(
        splits, PROJECT_ROOT / "data" / run_id, f"{s3_prefix}/data"
    )
    print(json.dumps({k: len(v) for k, v in splits.items()}), json.dumps(uris, indent=2))

    mlflow_cfg = cfg.get("mlflow") or {}
    arn = mlflow_cfg.get("tracking_server_arn")
    execution_id = f"local-{run_id}"
    summary = (
        tracking.get_or_create_run(arn, mlflow_cfg["experiment_name"], execution_id,
                                   {"pipeline_name": "scripts/run_sft.py",
                                    "base_model": cfg["base_model"]})
        if arn else None
    )
    if summary:
        tracking.log(arn, summary, params={"data": cfg["data"]}, metrics=sft.data_stats(splits),
                     tags={f"data_uri.{k}": v for k, v in uris.items()},
                     json_artifacts={f"data/{k}_sample.json": v[:5] for k, v in splits.items()})
        print(f"MLflow summary run: {summary} (experiment {mlflow_cfg['experiment_name']})")

    print("[2/4] Registering model package group and datasets")
    group = sft.ensure_model_package_group(cfg["model_package_group_name"])
    train_arn = sft.register_dataset(f"{cfg['model_package_group_name']}-train", uris["train"], role)
    val_arn = sft.register_dataset(f"{cfg['model_package_group_name']}-val", uris["validation"], role)
    print(f"group={group.model_package_group_arn}\ntrain={train_arn}\nvalidation={val_arn}")
    if summary:
        tracking.log(arn, summary, tags={"model_package_group": cfg["model_package_group_name"],
                                         "dataset_arn.train": train_arn,
                                         "dataset_arn.validation": val_arn})

    trainer = sft.build_trainer(
        base_model=cfg["base_model"],
        model_package_group=group,
        training_dataset=train_arn,
        validation_dataset=val_arn,
        s3_output_path=f"{s3_prefix}/output/",
        hyperparameters=cfg["hyperparameters"],
        role=role,
        mlflow_resource_arn=arn,
        mlflow_experiment_name=mlflow_cfg.get("experiment_name"),
        mlflow_run_name=f"sft-{execution_id}",
    )
    if summary:
        tracking.log(arn, summary, params={"base_model": cfg["base_model"],
                                           "hyperparameters": trainer.hyperparameters.to_dict()})
    print("Hyperparameters:", json.dumps(trainer.hyperparameters.to_dict(), indent=2))
    if args.dry_run:
        print("Dry run: not starting training.")
        if summary:
            tracking.end_run(arn, summary, "KILLED")
        return

    print("[3/4] Starting serverless SFT job")
    training_job = trainer.train(wait=not args.no_wait)
    result = sft.describe_result(training_job)
    print(json.dumps(result, indent=2, default=str))

    manifest_dir = PROJECT_ROOT / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / f"training-{result['training_job_name']}.json").write_text(
        json.dumps({**result, "data": uris, "s3_prefix": s3_prefix}, indent=2, default=str)
    )
    if summary and not args.no_wait:
        tracking.log(arn, summary, tags={"training_job_name": result["training_job_name"],
                                         "training_job_status": result["status"],
                                         "model_package_arn": result["model_package_arn"],
                                         "sft_mlflow_run_id": result["mlflow_run_id"]})
        tracking.log_training_job_logs(arn, summary, result["training_job_name"], "ServerlessSFT-job")
        if result["mlflow_run_id"]:
            tracking.client(arn).set_tag(result["mlflow_run_id"], tracking.EXECUTION_TAG, execution_id)
            tracking.copy_metric_history(arn, result["mlflow_run_id"], summary, ("train/", "perf/"))
    if args.no_wait or result["status"] != "Completed":
        if summary and not args.no_wait:
            tracking.end_run(arn, summary, "FAILED")
        return

    gate = cfg["quality_gate"]
    value = sft.final_metric(result["mlflow_resource_arn"], result["mlflow_run_id"], gate["metric"])
    passed = value <= gate["max_value"]
    if summary:
        tracking.log(arn, summary,
                     params={"quality_gate.metric": gate["metric"],
                             "quality_gate.max_value": gate["max_value"]},
                     metrics={"quality_gate/value": value, "quality_gate/passed": int(passed)},
                     tags={"quality_gate": "passed" if passed else "failed"})
        tracking.end_run(arn, summary, "FINISHED" if passed else "FAILED")
    print(f"[4/4] Quality gate: {gate['metric']}={value:.4f} (max {gate['max_value']}) -> "
          f"{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
