#!/usr/bin/env bash
# Run the CDK CLI for this app with the project venv.
#   ./cdk.sh synth | diff | deploy | destroy
# In SageMaker Studio /home is world-writable (mode 0777), which makes jsii refuse its
# package cache, so it is disabled here.
# cdk.json runs the app with .venv/bin/python explicitly: recent CDK CLI versions don't
# start the app with this script's PATH, so a bare `python3` would miss aws-cdk-lib.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv --system-site-packages .venv && .venv/bin/pip install -q -r requirements.txt; }
export PATH="$PWD/.venv/bin:$PATH"
export JSII_RUNTIME_PACKAGE_CACHE=disabled
export JSII_SILENCE_WARNING_DEPRECATED_NODE_VERSION=1
export AWS_SDK_UA_APP_ID="${AWS_SDK_UA_APP_ID:-AWSSkill-SageMaker}"
# Pinned: an unpinned CLI changed behavior mid-project (it stopped passing PATH to the app).
exec npx -y aws-cdk@2.1146.1 "$@"
