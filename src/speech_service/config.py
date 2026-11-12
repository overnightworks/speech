"""What varies by machine; everything else is a constant beside its owner."""

import tomllib
from pathlib import Path
from typing import Annotated, Self

from huggingface_hub import constants as huggingface_hub_constants
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from webauth.https_client import is_https_url

from speech_client.wire import VoiceId
from speech_service.hearing_models import HearingModel
from speech_service.voices import PIPER_VOICES

_PROVIDER_ROOT_ERROR = "provider root must be absolute"
_VOICES_ERROR = (
    "SPEECH_VOICES must list catalogue voice ids separated by commas, each at most once"
)
_CONFIG_FILE_ERROR = "config file must be absolute"
_KEY_SET_URL_ERROR = "jwks_url must be an https URL"
_SERVICE_GROUP_ERROR = "service_group must be a group path such as /services/name"
_APP_NAME_ERROR = "an app name must be one group path segment"
_SHARED_CLIENT_ERROR = "workload_client must belong to one app"
_CONFIG_FILE_UNREADABLE_ERROR = "SPEECH_CONFIG_FILE must name a readable file"


class Settings(BaseSettings):
    """Read from `SPEECH_*` in the environment."""

    model_config = SettingsConfigDict(env_prefix="SPEECH_", hide_input_in_errors=True)

    device: str = "cuda"
    voices: Annotated[frozenset[VoiceId], NoDecode] = frozenset(PIPER_VOICES)
    voice_cache: Path = Field(default_factory=lambda: Path.home() / ".cache" / "piper")
    huggingface_cache: Path = Field(
        default_factory=lambda: Path(huggingface_hub_constants.HF_HUB_CACHE)
    )
    provider_root: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    busy_wait_seconds: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    max_waiting_requests: int = Field(default=3, ge=0)
    max_body_bytes: int = Field(default=32_768, ge=1)
    max_input_characters: int = Field(default=4096, ge=1)
    hearing_model: HearingModel | None = None
    hearing_busy_wait_seconds: float = Field(default=30.0, ge=0, allow_inf_nan=False)
    hearing_max_waiting_requests: int = Field(default=3, ge=0)
    max_hearing_sessions: int = Field(default=1, ge=1)
    hearing_final_wait_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    max_recording_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    max_recording_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    config_file: Path | None = None

    @field_validator("config_file")
    @classmethod
    def _config_file_is_absolute(cls, path: Path | None) -> Path | None:
        if path is not None and not path.is_absolute():
            raise ValueError(_CONFIG_FILE_ERROR)
        return path

    @field_validator("provider_root")
    @classmethod
    def _provider_root_is_absolute(cls, directory: Path | None) -> Path | None:
        if directory is not None and not directory.is_absolute():
            raise ValueError(_PROVIDER_ROOT_ERROR)
        return directory

    @field_validator("voices", mode="before")
    @classmethod
    def _voices_are_distinct_catalogue_ids(
        cls, names: str | frozenset[VoiceId]
    ) -> frozenset[VoiceId]:
        if not isinstance(names, str):
            return frozenset(names)
        listed = (
            [_voice(name.strip()) for name in names.split(",")] if names.strip() else []
        )
        if len(set(listed)) != len(listed):
            raise ValueError(_VOICES_ERROR)
        return frozenset(listed)


def _voice(name: str) -> VoiceId:
    # Pydantic's enum error lists every catalogue voice; a fixed message does not.
    try:
        return VoiceId(name)
    except ValueError:
        raise ValueError(_VOICES_ERROR) from None


def load_settings() -> Settings:
    """Read `SPEECH_*` from the environment."""
    return Settings()


class _FileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, strict=True
    )


class IdentityConfig(_FileModel):
    """Who vouches for the callers, and the one group that may use this service."""

    issuer: str
    jwks_url: str = Field(repr=False)
    audience: str
    workload_role: str
    group_claim: str
    service_group: str

    @field_validator("jwks_url")
    @classmethod
    def _key_set_is_served_over_https(cls, url: str) -> str:
        if not is_https_url(url):
            raise ValueError(_KEY_SET_URL_ERROR)
        return url

    @field_validator("service_group")
    @classmethod
    def _service_group_is_a_group_path(cls, group: str) -> str:
        if not group.startswith("/") or "" in group[1:].split("/"):
            raise ValueError(_SERVICE_GROUP_ERROR)
        return group


class AppConfig(_FileModel):
    """An app the service admits: its workload client."""

    workload_client: str


class CallerConfig(_FileModel):
    """The service's trust in its callers, read from `SPEECH_CONFIG_FILE`."""

    identity: IdentityConfig
    apps: Annotated[dict[str, AppConfig], Field(min_length=1)]

    @field_validator("apps")
    @classmethod
    def _app_names_are_one_segment(
        cls, apps: dict[str, AppConfig]
    ) -> dict[str, AppConfig]:
        if any(not name or "/" in name for name in apps):
            raise ValueError(_APP_NAME_ERROR)
        return apps

    @model_validator(mode="after")
    def _each_client_belongs_to_one_app(self) -> Self:
        clients = [app.workload_client for app in self.apps.values()]
        if len(set(clients)) != len(clients):
            raise ValueError(_SHARED_CLIENT_ERROR)
        return self


def load_caller_config(path: Path) -> CallerConfig:
    """Read the caller configuration; a bad file fails naming a field, not a value."""
    try:
        text = path.read_text()
    except OSError:
        raise ValueError(_CONFIG_FILE_UNREADABLE_ERROR) from None
    return CallerConfig.model_validate(tomllib.loads(text))
