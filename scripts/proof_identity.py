"""The throwaway identity the stage proof speaks with.

The proof serves its own key set over local HTTPS, writes the service
configuration that trusts it, and hands the processes that speak a separate
minting file. Tokens are minted fresh for each request, because the proof
outlives a workload token's lifetime.
"""

from __future__ import annotations

import datetime as dt
import http.server
import ipaddress
import json
import os
import ssl
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Final

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

KEY_BITS: Final = 2048
WORKLOAD_SECONDS: Final = 300
PERSON_SECONDS: Final = 300
CERTIFICATE_DAYS: Final = 1
HANDSHAKE_TIMEOUT_SECONDS: Final = 5
ALGORITHM: Final = "RS256"
OWNER_ONLY_FILE: Final = 0o600
OWNER_ONLY_DIRECTORY: Final = 0o700
WORLD_READABLE_FILE: Final = 0o644
WORLD_READABLE_DIRECTORY: Final = 0o755
CA_NAME: Final = "proof CA"
CA_KEY_USAGE: Final = x509.KeyUsage(
    digital_signature=False, content_commitment=False, key_encipherment=False,
    data_encipherment=False, key_agreement=False, key_cert_sign=True,
    crl_sign=False, encipher_only=False, decipher_only=False,
)  # fmt: skip
APP: Final = "proof"
KEY_IDENTIFIER: Final = "proof-key"


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


@dataclass(frozen=True)
class ProofIdentity:
    """The files and URL a running proof identity offers."""

    configuration_path: Path
    minting_path: Path
    ca_path: Path
    jwks_url: str


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


def _write_private(path: Path, content: str | bytes) -> None:
    data = content.encode() if isinstance(content, str) else content
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, OWNER_ONLY_FILE)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _common_name(text: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, text)])


def _certificate_builder(
    subject: x509.Name, issuer: x509.Name, public_key: ec.EllipticCurvePublicKey
) -> x509.CertificateBuilder:
    now = dt.datetime.now(tz=dt.UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=CERTIFICATE_DAYS))
    )


def _self_signed_certificate(
    key: ec.EllipticCurvePrivateKey,
) -> x509.Certificate:
    name = _common_name("localhost")
    return (
        _certificate_builder(name, name, key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )


def _signed_certificate(
    subject: str,
    public_key: ec.EllipticCurvePublicKey,
    signer: ec.EllipticCurvePrivateKey,
    *,
    host: str | None,
) -> x509.Certificate:
    """A CA certificate when `host` is None, else a server certificate for `host`."""
    builder = (
        _certificate_builder(_common_name(subject), _common_name(CA_NAME), public_key)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False
        )
        .add_extension(
            x509.BasicConstraints(ca=host is None, path_length=None), critical=True
        )
    )
    if host is None:
        return builder.add_extension(CA_KEY_USAGE, critical=True).sign(
            signer, hashes.SHA256()
        )
    return (
        builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(signer.public_key()),
            critical=False,
        )
        .sign(signer, hashes.SHA256())
    )


def issue_certificates(directory: Path, host: str, key_set: bytes) -> None:
    """Write a test CA, a key server's certificate for `host`, and the key set.

    The files are readable by the non-root user of the container that mounts them.
    """
    authority_key = ec.generate_private_key(ec.SECP256R1())
    server_key = ec.generate_private_key(ec.SECP256R1())
    authority = _signed_certificate(
        CA_NAME, authority_key.public_key(), authority_key, host=None
    )
    server = _signed_certificate(
        host, server_key.public_key(), authority_key, host=host
    )
    pem = serialization.Encoding.PEM
    files = {
        directory / "ca" / "root_ca.crt": authority.public_bytes(pem),
        directory / "key-server" / "tls.crt": server.public_bytes(pem),
        directory / "key-server" / "tls.key": _private_pem(server_key),
        directory / "key-server" / "key-set.json": key_set,
    }
    for path, content in files.items():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        path.chmod(WORLD_READABLE_FILE)
        path.parent.chmod(WORLD_READABLE_DIRECTORY)


def fetch_key_set(url: str, trust: Path) -> bytes:
    """The key set the proof identity serves, read once from the host."""
    context = ssl.create_default_context(cafile=str(trust))
    with urllib.request.urlopen(url, context=context, timeout=10) as response:  # noqa: S310 - the proof's own https URL
        return response.read()


def _key_server(
    key_set: bytes, certificate_path: Path, key_path: Path
) -> http.server.ThreadingHTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = HANDSHAKE_TIMEOUT_SECONDS

        def do_GET(self) -> None:
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(key_set)))
            self.end_headers()
            self.wfile.write(key_set)

        def log_message(self, *_: object) -> None:
            """Log nothing."""

    class QuietServer(http.server.ThreadingHTTPServer):
        def handle_error(self, *_: object) -> None:
            """Report nothing: a client rejecting the certificate is expected."""

    server = QuietServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate_path, key_path)
    # The handshake runs on the request thread under its timeout, so a client
    # that withholds it cannot block the accept loop or the shutdown.
    server.socket = context.wrap_socket(
        server.socket, server_side=True, do_handshake_on_connect=False
    )
    return server


def _private_pem(key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _configuration_text(minting: Minting, jwks_url: str, voice: str) -> str:
    def value(text: str) -> str:
        return json.dumps(text)

    return "\n".join(
        [
            "[identity]",
            f"issuer = {value(minting.issuer)}",
            f"jwks_url = {value(jwks_url)}",
            f"audience = {value(minting.audience)}",
            f"workload_role = {value(minting.workload_role)}",
            f"group_claim = {value(minting.group_claim)}",
            f"service_group = {value(minting.service_group)}",
            "",
            f"[apps.{APP}]",
            f"workload_client = {value(minting.client_id)}",
            f"voices = [{value(voice)}]",
            "hear = true",
            "",
        ]
    )


@contextmanager
def proof_identity(directory: Path, voice: str) -> Iterator[ProofIdentity]:
    """Serve a throwaway key set over local HTTPS and write the files that use it.

    The proof app may hear and may name exactly `voice`, a catalogue id the
    caller passes in because this module imports nothing of the service.
    Everything lands in a private subdirectory of `directory`. The key server
    stops on exit; removing `directory` is the caller's.
    """
    identity_directory = directory / "identity"
    identity_directory.mkdir(mode=OWNER_ONLY_DIRECTORY)
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=KEY_BITS)
    tls_key = ec.generate_private_key(ec.SECP256R1())
    certificate_pem = _self_signed_certificate(tls_key).public_bytes(
        serialization.Encoding.PEM
    )
    ca_path = identity_directory / "ca.pem"
    tls_path = identity_directory / "tls-key.pem"
    _write_private(ca_path, certificate_pem)
    _write_private(tls_path, _private_pem(tls_key))
    public_jwk = json.loads(
        jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key())
    ) | {"kid": KEY_IDENTIFIER, "use": "sig", "alg": ALGORITHM}
    server = _key_server(json.dumps({"keys": [public_jwk]}).encode(), ca_path, tls_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"https://localhost:{server.server_address[1]}"
        minting = Minting(
            private_key_pem=_private_pem(signing_key).decode(),
            key_identifier=KEY_IDENTIFIER,
            issuer=f"{origin}/realm",
            audience="speech-proof",
            client_id="proof-app",
            workload_role="workload",
            group_claim="groups",
            app_group=f"/apps/{APP}",
            service_group="/services/speech",
            workload_subject="proof-service-account",
            person_subject="proof-person",
        )
        configuration_path = identity_directory / "speech.toml"
        minting_path = identity_directory / "minting.json"
        jwks_url = f"{origin}/certs"
        _write_private(
            configuration_path, _configuration_text(minting, jwks_url, voice)
        )
        _write_private(minting_path, json.dumps(asdict(minting)))
        yield ProofIdentity(configuration_path, minting_path, ca_path, jwks_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
