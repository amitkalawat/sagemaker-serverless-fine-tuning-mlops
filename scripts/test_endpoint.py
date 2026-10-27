"""Test the deployed model: compare the fine-tuned adapter with the base model on held-out data.

By default it uses the endpoint from config/pipeline.yaml, its current adapter component, and
the eval split recorded on the newest Approved model package, so a fresh deployment can be
checked with no arguments.

Usage:
  python3 scripts/test_endpoint.py                      # 5 held-out examples, base vs fine-tuned
  python3 scripts/test_endpoint.py -n 10 --skip 3       # more examples, skip the smoke-test ones
  python3 scripts/test_endpoint.py --prompt "Summarize: ..."   # ad-hoc prompt
  python3 scripts/test_endpoint.py --json results.json  # also save every response
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config  # noqa: E402
from pipeline.steps import deploy  # noqa: E402


def components(sm, endpoint: str):
    base, adapters = None, []
    for ic in sm.list_inference_components(
        EndpointNameEquals=endpoint, StatusEquals="InService"
    )["InferenceComponents"]:
        name = ic["InferenceComponentName"]
        spec = sm.describe_inference_component(InferenceComponentName=name)["Specification"]
        if spec.get("BaseInferenceComponentName"):
            adapters.append(name)
        else:
            base = name
    return base, adapters


def latest_eval_uri(sm, group: str) -> str:
    packages = sm.list_model_packages(
        ModelPackageGroupName=group, ModelApprovalStatus="Approved", SortBy="CreationTime",
        SortOrder="Descending", MaxResults=10,
    )["ModelPackageSummaryList"]
    for p in packages:
        meta = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"]).get(
            "CustomerMetadataProperties", {}
        )
        if meta.get("eval_data_uri"):
            return meta["eval_data_uri"]
    raise SystemExit("No Approved model package with eval_data_uri metadata; pass --eval-uri")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", help="default: deploy.endpoint_name")
    parser.add_argument("--adapter", help="adapter component (default: the endpoint's adapter)")
    parser.add_argument("--eval-uri", help="eval JSONL (default: newest Approved package's)")
    parser.add_argument("-n", type=int, default=5, help="number of examples")
    parser.add_argument("--skip", type=int, default=3, help="skip the first N (smoke-test) rows")
    parser.add_argument("--prompt", help="send one ad-hoc prompt instead of eval examples")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-base", action="store_true", help="only query the fine-tuned adapter")
    parser.add_argument("--json", help="write all responses to this file")
    args = parser.parse_args()

    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg["region"])
    endpoint = args.endpoint or cfg["deploy"]["endpoint_name"]
    base, adapters = components(sm, endpoint)
    adapter = args.adapter or (adapters[0] if len(adapters) == 1 else None)
    if not adapter:
        raise SystemExit(f"Pass --adapter; InService adapters on {endpoint}: {adapters}")
    targets = [("fine-tuned", adapter)] + ([] if args.no_base or not base else [("base", base)])
    print(f"Endpoint {endpoint}: fine-tuned={adapter} base={base}\n")

    if args.prompt:
        examples = [{"prompt": args.prompt, "completion": ""}]
    else:
        uri = args.eval_uri or latest_eval_uri(sm, cfg["stack"]["model_package_group_name"])
        examples = deploy.read_jsonl(uri, args.skip + args.n)[args.skip:]
        print(f"Eval data: {uri} (rows {args.skip + 1}-{args.skip + len(examples)})\n")

    results, stats = [], {label: [] for label, _ in targets}
    for i, ex in enumerate(examples, 1):
        print(f"=== {i}. {ex['prompt'].splitlines()[0][:150]}")
        if ex["completion"]:
            print(f"  reference : {ex['completion'][:300]}")
        row = {**ex}
        for label, component in targets:
            out = deploy.invoke(endpoint, component, ex["prompt"], args.max_tokens)
            tokens = out["usage"].get("completion_tokens", 0)
            stats[label].append((out["latency_s"], tokens))
            row[label] = out["text"]
            print(f"  {label:<10}: [{out['latency_s']:.1f}s, {tokens} tok] {out['text'][:300]!r}")
        results.append(row)
        print()

    for label, values in stats.items():
        print(f"{label:<10} mean latency {statistics.mean(v[0] for v in values):.2f}s, "
              f"mean completion tokens {statistics.mean(v[1] for v in values):.0f}")
    if any(r["completion"] for r in results):
        words = statistics.mean(len(r["completion"].split()) for r in results)
        print(f"reference  mean length {words:.0f} words")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
