"""Show where things stand: latest training and deploy executions, approvals, model versions,
and the endpoint.

Usage:
  python3 scripts/status.py            # names from config/pipeline.yaml (stack section)
  python3 scripts/status.py --last 3   # more executions per pipeline
"""

import argparse
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config  # noqa: E402


def executions(sm, pipeline_name: str, last: int):
    try:
        summaries = sm.list_pipeline_executions(
            PipelineName=pipeline_name, SortBy="CreationTime", SortOrder="Descending",
            MaxResults=last,
        )["PipelineExecutionSummaries"]
    except sm.exceptions.ResourceNotFound:
        print(f"  pipeline {pipeline_name} not found (deploy the CDK stack first)")
        return
    if not summaries:
        print("  no executions yet")
    for ex in summaries:
        arn = ex["PipelineExecutionArn"]
        print(f"  {ex['StartTime']:%Y-%m-%d %H:%M} {ex['PipelineExecutionStatus']:<10} "
              f"{ex.get('PipelineExecutionDisplayName', '')}  {arn.rsplit('/', 1)[-1]}")
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]
        for s in reversed(steps):
            reason = f"  <- {s['FailureReason'][:150]}" if s.get("FailureReason") else ""
            print(f"      {s['StepName']:<20} {s['StepStatus']}{reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", type=int, default=1, help="executions to show per pipeline")
    args = parser.parse_args()

    cfg = load_config()
    stack = cfg["stack"]
    sm = boto3.client("sagemaker", region_name=cfg["region"])

    print(f"Training pipeline {stack['pipeline_name']}")
    executions(sm, stack["pipeline_name"], args.last)
    print(f"\nDeploy pipeline {stack['deploy_pipeline_name']}")
    executions(sm, stack["deploy_pipeline_name"], args.last)

    print(f"\nModel package group {stack['model_package_group_name']} (newest first)")
    try:
        packages = sm.list_model_packages(
            ModelPackageGroupName=stack["model_package_group_name"], SortBy="CreationTime",
            SortOrder="Descending", MaxResults=5,
        )["ModelPackageSummaryList"]
        for p in packages:
            print(f"  v{p['ModelPackageVersion']:<3} {p.get('ModelApprovalStatus', '-'):<22} "
                  f"{p['CreationTime']:%Y-%m-%d %H:%M}")
    except sm.exceptions.ClientError as e:
        print(f"  {e}")

    endpoint = cfg["deploy"]["endpoint_name"]
    print(f"\nEndpoint {endpoint}")
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint)
        print(f"  {desc['EndpointStatus']} ({cfg['deploy']['instance_type']}, billed while it exists)")
        for ic in sm.list_inference_components(EndpointNameEquals=endpoint)["InferenceComponents"]:
            spec = sm.describe_inference_component(
                InferenceComponentName=ic["InferenceComponentName"]
            )["Specification"]
            kind = "adapter" if spec.get("BaseInferenceComponentName") else "base"
            print(f"  {kind:<8} {ic['InferenceComponentName']:<48} {ic['InferenceComponentStatus']}")
    except sm.exceptions.ClientError:
        print("  not deployed")


if __name__ == "__main__":
    main()
