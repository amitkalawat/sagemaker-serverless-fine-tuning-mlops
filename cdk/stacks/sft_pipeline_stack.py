"""CDK stack: the training and deploy pipelines and everything they need.

    training pipeline --(Succeeded event)--> deploy pipeline --(ManualApproval via SQS)-->
    SageMaker endpoint
"""

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from constructs import Construct


def names(cfg: dict, account: str, region: str) -> dict:
    """Physical names that must be known before deploy, because they're baked into the
    pipeline definition that the app generates at synth time."""
    stack_cfg = cfg["stack"]
    # `AmazonSageMakerModelCustomizationCoreAccess` only lets a role pass roles under
    # service-role/ whose name contains "SageMaker", and only reads/writes buckets whose name
    # contains "sagemaker". Naming to match keeps the managed policy usable as-is.
    role_name = f"SageMaker-{stack_cfg['pipeline_name']}-{region}"[:64]
    topic_name = f"{stack_cfg['pipeline_name']}-notifications"
    queue_name = f"{stack_cfg['deploy_pipeline_name']}-approvals"
    return {
        "pipeline_name": stack_cfg["pipeline_name"],
        "deploy_pipeline_name": stack_cfg["deploy_pipeline_name"],
        "topic_name": topic_name,
        "topic_arn": f"arn:aws:sns:{region}:{account}:{topic_name}",
        "queue_name": queue_name,
        "queue_url": f"https://sqs.{region}.amazonaws.com/{account}/{queue_name}",
        "model_package_group_name": stack_cfg["model_package_group_name"],
        "role_name": role_name,
        "role_arn": f"arn:aws:iam::{account}:role/service-role/{role_name}",
        "data_bucket": f"sagemaker-sft-pipeline-{account}-{region}",
    }


class SftPipelineStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: dict,
        resource_names: dict,
        artifact_bucket: str,
        pipeline_definition: str,
        deploy_pipeline_definition: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        n = resource_names

        data_bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=n["data_bucket"],
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            lifecycle_rules=[s3.LifecycleRule(noncurrent_version_expiration=Duration.days(30))],
            # Holds training data and the fine-tuned model artifacts the registry points at.
            removal_policy=RemovalPolicy.RETAIN,
        )

        role = iam.Role(
            self,
            "PipelineRole",
            role_name=n["role_name"],
            path="/service-role/",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Runs the serverless SFT pipeline, its @step jobs and the SFT job",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerModelCustomizationCoreAccess"
                )
            ],
        )
        data_bucket.grant_read_write(role)
        # The SDK uploads serialized @step code here at synth time; step jobs write results.
        artifact = s3.Bucket.from_bucket_name(self, "ArtifactBucket", artifact_bucket)
        artifact.grant_read_write(role, f"{n['pipeline_name']}/*")
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"], resources=[artifact.bucket_arn]
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DatasetHubAndTagging",
                actions=[
                    "sagemaker:CreateHub",
                    "sagemaker:DescribeHub",
                    "sagemaker:AddTags",
                    "sagemaker:ListTags",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:hub/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:hub-content/*",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:training-job/*",
                ],
            )
        )
        # The role passes itself to the jobs it starts (serverless SFT, dataset import). Some
        # SDK calls pass the path-less form of the ARN, which the managed policy's
        # service-role/*SageMaker* PassRole pattern doesn't match, so allow both forms.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="PassSelfToSageMaker",
                actions=["iam:PassRole"],
                resources=[n["role_arn"], f"arn:aws:iam::{self.account}:role/{n['role_name']}"],
                conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            )
        )
        # MLflow reads/writes and CloudWatch log reads (/aws/sagemaker/*) come from the managed
        # policy. Evaluate, RegisterCandidate and the deploy steps list the execution's steps
        # to find the step jobs whose logs they attach to the MLflow run.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ListOwnPipelineSteps",
                actions=["sagemaker:ListPipelineExecutionSteps"],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{n['pipeline_name']}",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{n['pipeline_name']}/*",
                ],
            )
        )

        group = sagemaker.CfnModelPackageGroup(
            self,
            "ModelPackageGroup",
            model_package_group_name=n["model_package_group_name"],
            model_package_group_description=f"Serverless SFT fine-tunes of {cfg['base_model']}",
        )
        # Deleting a group that still has model versions fails; keep it with the stack's data.
        group.apply_removal_policy(RemovalPolicy.RETAIN)

        pipeline = sagemaker.CfnPipeline(
            self,
            "Pipeline",
            pipeline_name=n["pipeline_name"],
            pipeline_description=f"Serverless SFT (LoRA) of {cfg['base_model']} with a quality gate",
            role_arn=role.role_arn,
            pipeline_definition={"PipelineDefinitionBody": pipeline_definition},
        )
        pipeline.node.add_dependency(role)
        pipeline.node.add_dependency(group)

        # Deploy pipeline: approval queue, endpoint permissions, the pipeline, its trigger.
        approval_queue = sqs.Queue(
            self,
            "ApprovalQueue",
            queue_name=n["queue_name"],
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            retention_period=Duration.days(14),
            visibility_timeout=Duration.seconds(30),
        )
        endpoint = cfg["deploy"]["endpoint_name"]
        # The Pipelines service posts the ManualApproval callback token with the pipeline role.
        approval_queue.grant_send_messages(role)
        role.add_to_policy(
            iam.PolicyStatement(
                sid="EndpointSmokeTestAndTagging",
                # DeleteModel: DeployEndpoint removes the base Model it built when the endpoint
                # already has a base component (not in the managed policy).
                actions=["sagemaker:InvokeEndpoint", "sagemaker:AddTags", "sagemaker:ListTags",
                         "sagemaker:DeleteModel"],
                # Scoped to resources named after the configured endpoint (ModelBuilder names
                # the config, components and models <endpoint>-...). Deploying to a different
                # EndpointName parameter needs a matching config change and redeploy.
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:{kind}/{endpoint}*"
                    for kind in ("endpoint", "endpoint-config", "inference-component", "model")
                ],
            )
        )
        deploy_pipeline_arn = (
            f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{n['deploy_pipeline_name']}"
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ListDeployPipelineSteps",
                actions=["sagemaker:ListPipelineExecutionSteps"],
                resources=[deploy_pipeline_arn, f"{deploy_pipeline_arn}/*"],
            )
        )
        deploy_pipeline = sagemaker.CfnPipeline(
            self,
            "DeployPipeline",
            pipeline_name=n["deploy_pipeline_name"],
            pipeline_description=(
                f"Approval-gated deployment of {cfg['base_model']} SFT models to a SageMaker endpoint"
            ),
            role_arn=role.role_arn,
            pipeline_definition={"PipelineDefinitionBody": deploy_pipeline_definition},
        )
        deploy_pipeline.node.add_dependency(role)
        deploy_pipeline.node.add_dependency(approval_queue)

        # Start the deploy pipeline whenever the training pipeline succeeds (it only succeeds
        # when the quality gate passed and RegisterCandidate ran).
        trigger_role = iam.Role(
            self,
            "DeployTriggerRole",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
            description="Lets EventBridge start the deploy pipeline",
        )
        trigger_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sagemaker:StartPipelineExecution"], resources=[deploy_pipeline_arn]
            )
        )
        training_pipeline_arn = (
            f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{n['pipeline_name']}"
        )
        trigger = events.CfnRule(
            self,
            "DeployOnTrainingSuccess",
            description="Start the deploy pipeline when the training pipeline succeeds",
            event_pattern={
                "source": ["aws.sagemaker"],
                "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
                "detail": {
                    "pipelineArn": [training_pipeline_arn],
                    "currentPipelineExecutionStatus": ["Succeeded"],
                },
            },
            targets=[
                events.CfnRule.TargetProperty(
                    id="DeployPipeline",
                    arn=deploy_pipeline_arn,
                    role_arn=trigger_role.role_arn,
                    sage_maker_pipeline_parameters=events.CfnRule.SageMakerPipelineParametersProperty(
                        pipeline_parameter_list=[]
                    ),
                )
            ],
        )
        trigger.node.add_dependency(deploy_pipeline)

        topic = sns.Topic(self, "PipelineNotifications", topic_name=n["topic_name"])
        if cfg["stack"].get("notification_email"):
            topic.add_subscription(subs.EmailSubscription(cfg["stack"]["notification_email"]))
        topic.grant_publish(role)  # ResolveModel publishes approval requests
        events.Rule(
            self,
            "PipelineExecutionStatus",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Building Pipeline Execution Status Change"],
                detail={
                    "pipelineArn": [training_pipeline_arn, deploy_pipeline_arn],
                    "currentPipelineExecutionStatus": ["Succeeded", "Failed", "Stopped"],
                },
            ),
            targets=[targets.SnsTopic(topic)],
        )

        CfnOutput(self, "PipelineName", value=n["pipeline_name"])
        CfnOutput(self, "PipelineRoleArn", value=role.role_arn)
        CfnOutput(self, "DataBucketName", value=data_bucket.bucket_name)
        CfnOutput(self, "ModelPackageGroupName", value=n["model_package_group_name"])
        CfnOutput(self, "NotificationTopicArn", value=topic.topic_arn)
        CfnOutput(self, "DeployPipelineName", value=n["deploy_pipeline_name"])
        CfnOutput(self, "ApprovalQueueUrl", value=approval_queue.queue_url)
        CfnOutput(self, "EndpointName", value=cfg["deploy"]["endpoint_name"])
