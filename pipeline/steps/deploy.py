"""Deploy a fine-tuned (LoRA) model package to a SageMaker real-time endpoint.

Layout on the endpoint (created by `ModelBuilder` for LoRA model customization):
    <endpoint>-inference-component   base Gemma 4 from JumpStart hosting artifacts (vLLM)
    <endpoint>-<execution id>        adapter component with the fine-tuned LoRA weights

Each deployment adds an adapter component next to the shared base component. After the smoke
test passes, older adapter components are deleted, so the endpoint serves one model version.
"""

import json
import time

import boto3

APPROVAL_STATUS_PENDING = "PendingManualApproval"
LATEST = "latest"  # ModelPackageArn value meaning "newest PendingManualApproval package"


def resolve_model_package(model_package_arn: str, model_package_group_name: str) -> dict:
    """Return details of `model_package_arn`, or of the newest pending candidate in the group
    when it is empty or `latest`."""
    sm = boto3.client("sagemaker")
    if not model_package_arn or model_package_arn == LATEST:
        packages = sm.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus=APPROVAL_STATUS_PENDING,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )["ModelPackageSummaryList"]
        if not packages:
            raise ValueError(
                f"No {APPROVAL_STATUS_PENDING} model package in group {model_package_group_name}"
            )
        model_package_arn = packages[0]["ModelPackageArn"]

    package = sm.describe_model_package(ModelPackageName=model_package_arn)
    metadata = package.get("CustomerMetadataProperties", {})
    container = package["InferenceSpecification"]["Containers"][0]
    training_job_name = metadata.get("training_job_name") or _job_name_from_s3_uri(
        container["ModelDataSource"]["S3DataSource"]["S3Uri"]
    )
    return {
        "model_package_arn": model_package_arn,
        "model_package_version": package.get("ModelPackageVersion"),
        "approval_status": package.get("ModelApprovalStatus"),
        "training_job_name": training_job_name,
        "base_model": container.get("BaseModel", {}).get("HubContentName"),
        "metadata": metadata,
    }


def _job_name_from_s3_uri(s3_uri: str) -> str:
    # .../<training job name>/output/model/
    parts = s3_uri.rstrip("/").split("/")
    return parts[parts.index("output", len(parts) - 3) - 1]


def set_approval(model_package_arn: str, status: str, description: str):
    boto3.client("sagemaker").update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus=status,
        ApprovalDescription=description[:1000],
    )


def deploy_adapter(
    training_job_name: str,
    endpoint_name: str,
    adapter_component_name: str,
    instance_type: str,
    role: str,
    accept_eula: bool,
    compute: dict,
) -> dict:
    """Deploy the LoRA adapter (and, on first deploy, the endpoint and base component).

    `compute` ({num_accelerators, num_cpus, memory_mb}) sizes the base component.
    """
    from sagemaker.core.inference_config import ResourceRequirements
    from sagemaker.core.resources import TrainingJob
    from sagemaker.serve import ModelBuilder

    # Pass the TrainingJob, not the ModelPackage: ModelBuilder only resolves the adapter
    # location (<model artifacts>/checkpoints/hf/) for TrainingJob/ModelTrainer inputs.
    builder = ModelBuilder(
        model=TrainingJob.get(training_job_name=training_job_name),
        role_arn=role,
        instance_type=instance_type,
    )
    builder.accept_eula = accept_eula
    built_model = f"{endpoint_name}-base-{int(time.time())}"
    builder.build(model_name=built_model)
    builder.deploy(
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=1,
        inference_component_name=adapter_component_name,
        inference_config=ResourceRequirements(
            requests={
                "num_accelerators": int(compute["num_accelerators"]),
                "num_cpus": int(compute["num_cpus"]),
                "memory": int(compute["memory_mb"]),
            }
        ),
    )
    wait_for_component(adapter_component_name)
    _delete_model_if_unused(built_model, adapter_component_name)
    return {"endpoint_name": endpoint_name, "inference_component_name": adapter_component_name}


def _delete_model_if_unused(model_name: str, adapter_component_name: str):
    """build() always creates a base Model, but deploy() reuses an existing base component
    when there is one; delete the new Model in that case so they don't pile up."""
    sm = boto3.client("sagemaker")
    spec = sm.describe_inference_component(InferenceComponentName=adapter_component_name)[
        "Specification"
    ]
    base = spec.get("BaseInferenceComponentName")
    base_model = (
        sm.describe_inference_component(InferenceComponentName=base)["Specification"].get("ModelName")
        if base else None
    )
    if base_model and base_model != model_name:
        sm.delete_model(ModelName=model_name)


def wait_for_component(name: str, timeout_s: int = 3600, poll_s: int = 30):
    sm = boto3.client("sagemaker")
    deadline = time.time() + timeout_s
    while True:
        desc = sm.describe_inference_component(InferenceComponentName=name)
        status = desc["InferenceComponentStatus"]
        if status == "InService":
            return
        if status == "Failed":
            raise RuntimeError(f"Inference component {name} failed: {desc.get('FailureReason')}")
        if time.time() > deadline:
            raise TimeoutError(f"Inference component {name} still {status} after {timeout_s}s")
        time.sleep(poll_s)


def read_jsonl(s3_uri: str, limit: int) -> list:
    bucket, _, key = s3_uri.removeprefix("s3://").partition("/")
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()][:limit]


def invoke(endpoint_name: str, component_name: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send one chat request to the adapter component and return text + latency."""
    runtime = boto3.client("sagemaker-runtime")
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    start = time.time()
    resp = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        InferenceComponentName=component_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    latency = time.time() - start
    body = json.loads(resp["Body"].read())
    text = body["choices"][0]["message"]["content"]
    return {"text": text, "latency_s": latency, "usage": body.get("usage", {})}


def smoke_test(endpoint_name: str, component_name: str, examples: list) -> dict:
    """Invoke the component on held-out examples. Raises if any request fails or is empty."""
    results = []
    for example in examples:
        out = invoke(endpoint_name, component_name, example["prompt"])
        if not out["text"].strip():
            raise RuntimeError(f"Empty completion for prompt: {example['prompt'][:200]}")
        results.append({**example, "response": out["text"], "latency_s": out["latency_s"],
                        "usage": out["usage"]})
    latencies = sorted(r["latency_s"] for r in results)
    return {
        "results": results,
        "metrics": {
            "smoke_test/requests": len(results),
            "smoke_test/latency_s_mean": sum(latencies) / len(latencies),
            "smoke_test/latency_s_max": latencies[-1],
            "smoke_test/completion_tokens_mean": sum(
                r["usage"].get("completion_tokens", 0) for r in results
            ) / len(results),
        },
    }


def remove_other_adapters(endpoint_name: str, keep: str) -> list:
    """Delete adapter components on the endpoint other than `keep` (base component stays)."""
    sm = boto3.client("sagemaker")
    removed = []
    paginator = sm.get_paginator("list_inference_components")
    for page in paginator.paginate(EndpointNameEquals=endpoint_name):
        for ic in page["InferenceComponents"]:
            name = ic["InferenceComponentName"]
            if name == keep:
                continue
            spec = sm.describe_inference_component(InferenceComponentName=name)["Specification"]
            if spec.get("BaseInferenceComponentName"):
                sm.delete_inference_component(InferenceComponentName=name)
                removed.append(name)
    return removed
