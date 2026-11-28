"""What the proof's own processes mint tokens from.

The clients read the minting file a proof identity wrote and mint tokens fresh
for each request, because the proof outlives a workload token's lifetime. This
module imports no `speech_service`, so a client loads it alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import jwt

from speech_client.hearing import SessionTokens

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

WORKLOAD_SECONDS: Final = 300
PERSON_SECONDS: Final = 300
ALGORITHM: Final = "RS256"


@dataclass(frozen=True)
class Minting:
    """What the proof's own processes mint tokens from; the service never reads it."""

    private_key_pem: str = field(repr=False)
    key_identifier: str
    issuer: str
    audience: str
    client_id: str
    workload_role: str
    group_claim: str
    app_group: str
    service_group: str
    workload_subject: str
    person_subject: str


@dataclass(frozen=True)
class TokenPair:
    """The app's workload token and the person's token for one request."""

    workload: str = field(repr=False)
    person: str = field(repr=False)

    async def tokens(self) -> SessionTokens:  # NOSONAR(S7503) async by contract
        """This pair as the token source of a hearing session."""
        return SessionTokens(self.workload, self.person)


def mint_pair(minting: Minting, clock: Callable[[], float] = time.time) -> TokenPair:
    """Mint a workload and a person token that are valid at `clock()`."""
    issued_at = int(clock())
    common = {
        "iss": minting.issuer,
        "aud": minting.audience,
        "azp": minting.client_id,
        "iat": issued_at,
    }
    workload = {
        **common,
        "exp": issued_at + WORKLOAD_SECONDS,
        "client_id": minting.client_id,
        "sub": minting.workload_subject,
        "realm_access": {"roles": [minting.workload_role]},
    }
    person = {
        **common,
        "exp": issued_at + PERSON_SECONDS,
        "sub": minting.person_subject,
        minting.group_claim: [minting.app_group, minting.service_group],
    }
    headers = {"kid": minting.key_identifier}
    return TokenPair(
        workload=jwt.encode(
            workload, minting.private_key_pem, ALGORITHM, headers=headers
        ),
        person=jwt.encode(person, minting.private_key_pem, ALGORITHM, headers=headers),
    )


def load_minting(path: Path) -> Minting:
    """Read the minting file a proof identity wrote."""
    return Minting(**json.loads(path.read_text()))
