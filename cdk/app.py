"""CDK app for the serverless SFT pipeline.

At synth time this builds the pipeline definition with the SageMaker Python SDK (v3), which
uploads the serialized @step functions to the SageMaker default bucket, and embeds the JSON
in AWS::SageMaker::Pipeline resources (training and deploy). Synth therefore needs AWS
credentials and the same Python minor version as the step image (3.12, SageMaker
Distribution 4.x).
"""

import os
import sys
from pathlib import Path

import aws_cdk as cdk
import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config  # noqa: E402
from pipeline.definition import build_pipeline  # noqa: E402
from pipeline.deploy_definition import build_deploy_pipeline  # noqa: E402
from sagemaker.core.helper.session_helper import Session  # noqa: E402
from stacks.sft_pipeline_stack import SftPipelineStack, names  # noqa: E402

cfg = load_config()
region = os.environ.get("CDK_DEFAULT_REGION") or cfg["region"]
account = (
    os.environ.get("CDK_DEFAULT_ACCOUNT")
    or boto3.client("sts", region_name=region).get_caller_identity()["Account"]
)
resource_names = names(cfg, account, region)
artifact_bucket = Session(boto3.Session(region_name=region)).default_bucket()

pipeline = build_pipeline(
    cfg,
    role_arn=resource_names["role_arn"],
    data_bucket=resource_names["data_bucket"],
    artifact_bucket=artifact_bucket,
    model_package_group_name=resource_names["model_package_group_name"],
    pipeline_name=resource_names["pipeline_name"],
)

deploy_pipeline = build_deploy_pipeline(
    cfg,
    role_arn=resource_names["role_arn"],
    artifact_bucket=artifact_bucket,
    approval_queue_url=resource_names["queue_url"],
    topic_arn=resource_names["topic_arn"],
    model_package_group_name=resource_names["model_package_group_name"],
    pipeline_name=resource_names["deploy_pipeline_name"],
)

app = cdk.App()
SftPipelineStack(
    app,
    cfg["stack"]["stack_name"],
    cfg=cfg,
    resource_names=resource_names,
    artifact_bucket=artifact_bucket,
    pipeline_definition=pipeline.definition(),
    deploy_pipeline_definition=deploy_pipeline.definition(),
    env=cdk.Environment(account=account, region=region),
)
app.synth()
