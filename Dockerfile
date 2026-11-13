# Both base images are held by digest, because a tag is a name its owner may
# move; Dependabot's docker entry moves the digest through review, and it reads
# only FROM lines, so uv is a stage of its own. The last stage is the small
# image, the default target; the gpu image is the `gpu` target.

# The voices the gpu image bakes and offers: every voice of the catalogue.
ARG GPU_VOICES=de_DE-thorsten-medium,de_DE-kerstin-medium,en_US-norman-medium,en_US-ljspeech-medium,en_GB-cori-medium,ryan
FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv
FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS python-base
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app

# uv and git exist only in this stage, because uv needs git to fetch the
# git-pinned webauth; the final stage copies the environment and carries neither.
FROM python-base AS dependencies
COPY --from=uv /uv /usr/local/bin/uv
# The interpreter is the image's own, so uv never downloads another one.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1
RUN apt-get update \
 && apt-get install --yes --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
COPY packages/client/pyproject.toml packages/client/pyproject.toml
COPY providers/provider-contract/pyproject.toml providers/provider-contract/pyproject.toml
# Third-party packages are wheel-only: --no-build refuses a source distribution,
# whose build script would run third-party code during the image build. The
# repository's own distributions and the git-pinned webauth are sources and
# build in the next step, with the exact build tools their pyprojects pin.
RUN uv sync --locked --no-dev --no-editable --no-install-project --no-build \
    --no-install-package overnightworks-webauth \
    --no-install-package overnightworks-speech-client \
    --no-install-package overnightworks-speech-provider-contract
COPY packages/client/src packages/client/src
COPY providers/provider-contract/src providers/provider-contract/src
COPY src src
RUN uv sync --locked --no-dev --no-editable

# The voices and the hearing model are fetched here, each at its pinned
# revision (Piper's voices also checked against their hashes). The Hub's own cache
# and logs stay in this stage.
FROM dependencies AS weights
ENV HF_HOME=/tmp/huggingface \
    SPEECH_VOICE_CACHE=/opt/voices \
    SPEECH_HUGGINGFACE_CACHE=/opt/hearing
RUN speech-fetch-voice && speech-fetch-hearing-model small

# The gpu image: the same service on the card, with the Qwen voice and the large
# hearing model. The service venv gains the CUDA 12 libraries CTranslate2 loads
# (the gpu extra); speech_service.cuda_libs finds them in that venv.
FROM dependencies AS gpu-dependencies
RUN uv sync --locked --no-dev --no-editable --extra gpu

# Every offered voice and large-v3, each at its pinned revision. Qwen's files and
# the hearing model share the Hub cache.
FROM dependencies AS gpu-weights
ARG GPU_VOICES
ENV HF_HOME=/tmp/huggingface \
    SPEECH_VOICE_CACHE=/opt/voices \
    SPEECH_HUGGINGFACE_CACHE=/opt/hearing \
    SPEECH_VOICES=$GPU_VOICES
RUN speech-fetch-voice && speech-fetch-hearing-model large-v3

# The Qwen provider is its own project on CPython 3.12, which vLLM-Omni caps
# below 3.14. uv installs that interpreter here, and the uv image's digest fixes
# which build of it. The environment is complete (CUDA 13 libraries and
# FlashInfer included) and is split into two layers in the next stage.
FROM python-base AS qwen-environment
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/providers/qwen/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1
RUN apt-get update \
 && apt-get install --yes --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build/providers/qwen
COPY providers/qwen/pyproject.toml providers/qwen/uv.lock ./
COPY providers/provider-contract/pyproject.toml ../provider-contract/pyproject.toml
# Wheel-only for third-party packages, as in the service's environment.
RUN uv sync --locked --no-dev --no-editable --no-install-project --no-build \
    --no-install-package overnightworks-speech-provider-contract
COPY providers/provider-contract/src ../provider-contract/src
COPY providers/qwen/src src
# NVIDIA's libraries are the largest part of the environment, and a layer stays
# under 10 GB: they move out, so the next stage copies them as a layer of their
# own and the rest of the environment as another.
RUN uv sync --locked --no-dev --no-editable \
 && mv /opt/providers/qwen/.venv/lib/python3.12/site-packages/nvidia /opt/qwen-nvidia

FROM python-base AS gpu
ARG GPU_VOICES
COPY --from=gpu-dependencies /opt/venv /opt/venv
COPY --from=gpu-weights /opt/voices /opt/voices
COPY --from=gpu-weights /opt/hearing /opt/hearing
COPY --from=qwen-environment /opt/python /opt/python
COPY --from=qwen-environment /opt/qwen-nvidia /opt/providers/qwen/.venv/lib/python3.12/site-packages/nvidia
COPY --from=qwen-environment /opt/providers/qwen/.venv /opt/providers/qwen/.venv
COPY deploy/speech.toml deploy/speech.toml
ENV SPEECH_VOICES=$GPU_VOICES \
    SPEECH_VOICE_CACHE=/opt/voices \
    SPEECH_HUGGINGFACE_CACHE=/opt/hearing \
    SPEECH_PROVIDER_ROOT=/opt/providers \
    SPEECH_LOAD_GPU_VOICES_AT_START=true \
    SPEECH_HEARING_MODEL=large-v3 \
    SPEECH_DEVICE=cuda
RUN useradd --no-create-home --uid 1001 speech
USER 1001

CMD ["speech-service"]

FROM python-base
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=weights /opt/voices /opt/voices
COPY --from=weights /opt/hearing /opt/hearing
COPY deploy/speech.toml deploy/speech.toml
RUN useradd --no-create-home --uid 1001 speech
USER 1001

CMD ["speech-service"]
