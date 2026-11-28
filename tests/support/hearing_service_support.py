"""The hearing proof's `Service` against a runtime served over TCP."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from tests.support.identity_support import callers_trusting
from tests.support.script_support import script_module
from tests.support.tcp_server_support import serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from proof_identity import ProofIdentity
    from proof_minting import Minting
    from speech_service.service import Runtime

CLIENT = script_module("hearing_client")


@asynccontextmanager
async def service_of(
    runtime: Runtime, minting: Minting, identity: ProofIdentity
) -> AsyncIterator[object]:
    """The proof's `Service` for `runtime`, served to callers trusting `minting`."""
    callers = callers_trusting(identity.configuration_path, minting)
    async with serving_over_tcp(runtime, callers) as server:
        yield CLIENT.Service(server.base_url, minting, None)
