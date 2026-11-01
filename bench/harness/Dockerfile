# The harness image: the official Python base image, git, an unprivileged user for the agent's commands, and
# mini-swe-agent with its dependencies in a virtual environment at /opt/harness, installed only from the
# hash-checked lock file. Each repository's bench image is built on top of this one.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates util-linux \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1001 --shell /bin/bash agent \
 && mkdir -p /testbed /out && chown agent:agent /testbed && chmod 700 /out
COPY requirements.lock /opt/harness-requirements.lock
RUN python -m venv /opt/harness \
 && /opt/harness/bin/pip install --no-cache-dir --require-hashes --no-deps -r /opt/harness-requirements.lock \
 && /opt/harness/bin/pip check
# No package may run code on every Python start. Fail the build if any .pth file other than the
# standard library's own appears in the harness environment.
RUN found="$(find /opt/harness -name '*.pth' ! -name 'distutils-precedence.pth')" \
 && if [ -n "$found" ]; then echo "unexpected .pth files: $found"; exit 1; fi
# LiteLLM would otherwise try to download its model price list at import; the bench counts cost itself.
ENV LITELLM_LOCAL_MODEL_COST_MAP=True \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /testbed
