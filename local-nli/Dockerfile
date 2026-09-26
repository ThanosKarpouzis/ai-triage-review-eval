# Local, offline zero-shot classifier for the Jev study.
# The build is the only step with network access; the scoring run has none (see compose.yaml).
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

# PyTorch, CPU-only build, from PyTorch's own index (no NVIDIA GPU libraries).
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0"
# Everything else, exact versions from PyPI, nothing beyond this list.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-deps -r /tmp/requirements.lock && pip check && pip freeze > /app-freeze.txt

# The model, pinned to one exact commit of the Hugging Face repository. MIT licence.
ARG MODEL_REVISION=cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download( \
      'MoritzLaurer/deberta-v3-large-zeroshot-v2.0', revision='${MODEL_REVISION}', local_dir='/models/deberta', \
      allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', \
                      'special_tokens_map.json', 'added_tokens.json', 'spm.model'])" \
 && rm -rf /models/deberta/.cache

# Make the model readable by any user (the downloader may write owner-only files). Kept as its own step
# so the cached download above is reused rather than fetched again.
RUN chmod -R a+rX /models

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin scorer
# /app is created explicitly with normal folder permissions, and the script is made world-readable,
# since the vault's files are owner-only and the run uses a different user.
RUN install -d -m 0755 /app
COPY --chmod=0755 score_nli.py /app/score_nli.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf HOME=/tmp NLI_MODEL_REVISION=${MODEL_REVISION}
USER 10001
ENTRYPOINT ["python", "/app/score_nli.py"]
