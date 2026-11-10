# speech

## Commands

Run each in its project's own directory (the root, or `providers/qwen`).

- `uv run --locked ruff check` and `uv run --locked ruff format --check`: lint and format.
- `uv run --locked pytest -q <named tests>`: targeted tests. The default run deselects the `keycloak` and `browser` markers.
- `uv run --locked python scripts/generate_browser_wire.py`: rewrite `packages/browser/src/hop.ts` from `PcmFormat` and `speech_client.relay`; `--check` fails on a drifted or missing module.
- In `packages/browser`: `pnpm install --frozen-lockfile`, `pnpm run check`, `pnpm run build` and `pnpm run test:coverage`.
- `uv run --locked pytest -q -n 2 -m browser tests/test_wire_route_browser_capture.py`: the browser capture flow in Chromium against `packages/browser/dist` (build it first; `uv run --locked playwright install chromium` once), under `/tmp/probe-stack.lock`.
- `uv run --locked --project ../qwen ruff check --config pyproject.toml .`: lint `providers/provider-contract`, from its directory.
- `uv run --locked pytest -q -m keycloak tests/test_callers_keycloak.py`: token shapes against a Keycloak you start yourself on a free port and name in `KEYCLOAK_URL`, under `/tmp/probe-stack.lock`.
- `uv run python scripts/prove_speech.py`: stage proof of health and speaking; starts the providers and takes `/tmp/probe-stack.lock` itself.

## Rules

- `providers/provider-contract` has no ruff and no lock of its own: never run `uv run ruff` there, it leaves a stray `uv.lock`.
- A new mandatory CI job is added by its display name to `mandatory_jobs` in `quality/config.toml`; the required check is `quality / verdict`.
- A new root entry is declared in `quality/config.toml` with a reason, or lives in its owner's directory.
- An advisory no relock reaches is allowlisted in `quality/config.toml` with its owning item and the model-library cap that holds it.
- `sonar-project.properties` carries only the keys the quality gate allows.
- Coverage reports are `coverage-<job>.xml` with paths relative to the repository root, as `[tool.coverage.run]` in the root `pyproject.toml` sets, and the browser package's `coverage-<job>.lcov`, with paths from the repository root too.
- The GPU providers are live code: do not delete them as unused.

## Owners

- `packages/client` owns the wire's types.
- `docs/wire.md` owns the wire's ruled behaviour.

## Documents

- `README.md`: read when you start or configure the service.
- `docs/VISION.md`: read when you judge whether a change still serves the service's purpose.
- `docs/ARCHITECTURE.md`: read when you change a building block, the trust model or delivery.
- `docs/wire.md`: read when you change or call the speak, hear or live-hearing wire.
