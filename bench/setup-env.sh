#!/usr/bin/env bash
# One-time setup of the bench's container side, run from the kit folder: bash bench/setup-env.sh
# Nothing from PyPI is installed on this machine. The harness (mini-swe-agent, MIT, with LiteLLM and its other
# dependencies) lives only inside the bench-harness image, installed from a hash-checked lock file.
#   1. writes bench/harness/requirements.lock, unless it exists (lock.py; pip runs in a throwaway container)
#   2. builds the bench-harness image from the official python:3.12-slim image at a recorded digest
#   3. creates bench-internal, a Docker network with no route out, and starts bench-egress, a small proxy that
#      is the network's only way out and allows the model gateway alone
#   4. checks that the isolation holds, with no model call and no spend
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H=bench/harness
TAG=bench-harness:2.4.6
GATEWAY=ai-gateway.vercel.sh:443

docker version --format 'docker {{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})' || { echo "docker not found; start OrbStack"; exit 1; }
if [ ! -f "$H/requirements.lock" ]; then python3 "$H/lock.py" || exit 1; fi
BASE="$(cat "$H/base-image.txt")"
echo "building $TAG on $BASE"
docker build --quiet --build-arg BASE_IMAGE="$BASE" -t "$TAG" "$H" >/dev/null || exit 1
docker network inspect bench-internal >/dev/null 2>&1 || docker network create --internal bench-internal >/dev/null
docker rm -f bench-egress >/dev/null 2>&1
docker run -d --name bench-egress --network bridge -v "$PWD/$H/egress_proxy.py:/p/egress_proxy.py:ro" \
  "$BASE" python /p/egress_proxy.py --allow "$GATEWAY" --port 3128 >/dev/null || exit 1
docker network connect bench-internal bench-egress || exit 1
sleep 2

pass=0; total=0
check() {  # check "description" expect(ok|fail) command...
  local what="$1" expect="$2"; shift 2; total=$((total+1))
  if "$@" >/dev/null 2>&1; then got=ok; else got=fail; fi
  if [ "$got" = "$expect" ]; then pass=$((pass+1)); echo "PASS  $what"; else echo "FAIL  $what"; fi
}
RUN=(docker run --rm --network bench-internal "$TAG")
PROXY=(-e HTTPS_PROXY=http://bench-egress:3128)
URL='import sys,urllib.request,urllib.error
try: urllib.request.urlopen(sys.argv[1], timeout=10)
except urllib.error.HTTPError: pass'
check "no direct route out of bench-internal" fail "${RUN[@]}" python -c "$URL" https://pypi.org/simple/
check "the proxy refuses other hosts" fail docker run --rm --network bench-internal "${PROXY[@]}" "$TAG" python -c "$URL" https://pypi.org/simple/
check "the proxy reaches the model gateway" ok docker run --rm --network bench-internal "${PROXY[@]}" "$TAG" python -c "$URL" https://ai-gateway.vercel.sh/v1/models
check "the agent user exists and runuser works (control for the next two)" ok docker run --rm --network none "$TAG" bash -c 'test "$(env -i PATH=/usr/bin:/bin /usr/sbin/runuser -u agent -- id -u)" = 1001'
check "the agent user can write the checkout folder (control)" ok docker run --rm --network none "$TAG" /usr/sbin/runuser -u agent -- touch /testbed/x
check "the agent user cannot read the harness's environment" fail docker run --rm --network none -e CANARY=not-for-the-agent "$TAG" bash -c '/usr/sbin/runuser -u agent -- cat /proc/1/environ'
check "the agent user cannot write the output folder" fail docker run --rm --network none "$TAG" /usr/sbin/runuser -u agent -- touch /out/x
check "the harness imports" ok docker run --rm --network none "$TAG" /opt/harness/bin/python -c 'import minisweagent, litellm'
echo "checks: $pass of $total passed"
docker run --rm --network none "$TAG" /opt/harness/bin/python -c '
from importlib.metadata import version as v
print("versions: python", __import__("sys").version.split()[0], "| mini-swe-agent", v("mini-swe-agent"), "| litellm", v("litellm"))'
echo "image $(docker image inspect "$TAG" --format '{{.Id}}')"
echo "proxy log (bench-egress):"; docker logs bench-egress 2>&1 | tail -5
