"""Approve or reject a deploy pipeline execution that is waiting in its ManualApproval step.

The ManualApproval CallbackStep posts a message with a callback token to the approval SQS
queue (stack output ApprovalQueueUrl). This script reads it and resumes the execution.

Usage:
  python3 scripts/approve_deployment.py --queue-url URL --list
  python3 scripts/approve_deployment.py --queue-url URL --approve [--execution-arn ARN]
  python3 scripts/approve_deployment.py --queue-url URL --reject --reason "why" [--execution-arn ARN]

Without --execution-arn it acts on the single waiting request (and refuses if there are
several). The approver recorded on the model package is the caller's IAM identity; a
rejection also sets the model package to Rejected.
"""

import argparse
import json
import sys

import boto3


def pending(queue_url: str) -> list:
    """Read waiting approval requests without consuming them (they reappear after 30s)."""
    sqs = boto3.client("sqs")
    seen, requests = set(), []
    for _ in range(5):
        msgs = sqs.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=10, VisibilityTimeout=30, WaitTimeSeconds=2
        ).get("Messages", [])
        for m in msgs:
            body = json.loads(m["Body"])
            if body["token"] in seen:
                continue
            seen.add(body["token"])
            requests.append({**body, "receipt_handle": m["ReceiptHandle"]})
        if not msgs:
            break
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue-url", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--approve", action="store_true")
    action.add_argument("--reject", action="store_true")
    parser.add_argument("--execution-arn", help="deploy pipeline execution to act on")
    parser.add_argument("--reason", default="", help="rejection reason")
    args = parser.parse_args()

    sm = boto3.client("sagemaker")
    requests = pending(args.queue_url)
    live = []
    for r in requests:
        status = sm.describe_pipeline_execution(
            PipelineExecutionArn=r["pipelineExecutionArn"]
        )["PipelineExecutionStatus"]
        if status == "Executing":
            live.append(r)
        else:  # stale request from a finished/stopped execution
            boto3.client("sqs").delete_message(QueueUrl=args.queue_url, ReceiptHandle=r["receipt_handle"])

    if args.list:
        for r in live:
            print(json.dumps({"pipelineExecutionArn": r["pipelineExecutionArn"],
                              "arguments": r.get("arguments", {})}, indent=2))
        print(f"{len(live)} waiting approval request(s)")
        return

    if args.execution_arn:
        live = [r for r in live if r["pipelineExecutionArn"] == args.execution_arn]
    if len(live) != 1:
        sys.exit(f"Expected exactly 1 matching waiting request, found {len(live)}; use --list "
                 "and pass --execution-arn")
    request = live[0]
    approver = boto3.client("sts").get_caller_identity()["Arn"]
    if args.approve:
        sm.send_pipeline_execution_step_success(
            CallbackToken=request["token"],
            OutputParameters=[{"Name": "approver", "Value": approver}],
        )
        print(f"Approved {request['pipelineExecutionArn']} as {approver}")
    else:
        if not args.reason:
            sys.exit("--reason is required with --reject")
        sm.send_pipeline_execution_step_failure(
            CallbackToken=request["token"], FailureReason=f"Rejected by {approver}: {args.reason}"
        )
        package_arn = request.get("arguments", {}).get("model_package_arn")
        if package_arn:
            sm.update_model_package(ModelPackageArn=package_arn, ModelApprovalStatus="Rejected",
                                    ApprovalDescription=f"Rejected by {approver}: {args.reason}"[:1000])
        print(f"Rejected {request['pipelineExecutionArn']}")
    boto3.client("sqs").delete_message(QueueUrl=args.queue_url, ReceiptHandle=request["receipt_handle"])


if __name__ == "__main__":
    main()
