# Both images are held by digest, because a tag is a name its owner may move;
# Dependabot's docker entry moves the digest through review, and it reads only
# FROM lines, so uv is a stage of its own.
FROM ghcr.io/astral-sh/uv:0.10.9@sha256:10902f58a1606787602f303954cea099626a4adb02acbac4c69920fe9d278f82 AS uv
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
    --no-install-package overnightworks-speech-runtime-client \
    --no-install-package overnightworks-speech-runtime-provider-contract
COPY packages/client/src packages/client/src
COPY providers/provider-contract/src providers/provider-contract/src
COPY src src
RUN uv sync --locked --no-dev --no-editable

# The voice and the hearing model are fetched here, each at its pinned revision
# (the voice also checked against its hashes); the Hub's own cache and logs stay
# in this stage.
FROM dependencies AS weights
ENV HF_HOME=/tmp/huggingface \
    SPEECH_VOICE_CACHE=/opt/voices \
    SPEECH_HUGGINGFACE_CACHE=/opt/hearing
RUN speech-runtime-fetch-voice de_DE-thorsten-medium \
 && speech-runtime-fetch-hearing-model small

FROM python-base
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=weights /opt/voices /opt/voices
COPY --from=weights /opt/hearing /opt/hearing
COPY deploy/speech.toml deploy/speech.toml
RUN useradd --no-create-home --uid 1001 speech
USER speech

CMD ["speech-runtime-service"]
