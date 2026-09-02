"""Shared, lazy-safe Azure OpenAI v1 client construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

from app.core.config import Settings, normalize_azure_openai_v1_base_url
from app.core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class AzureOpenAIClientHandle:
    """Keep both the SDK client and any Entra credential alive."""

    client: OpenAI
    credential: DefaultAzureCredential | None


def build_azure_openai_client(settings: Settings) -> AzureOpenAIClientHandle:
    """Construct one SDK client without logging credentials or endpoint text."""

    try:
        base_url = normalize_azure_openai_v1_base_url(settings.azure_openai_base_url or "")
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    credential: DefaultAzureCredential | None = None
    if settings.azure_openai_auth_mode == "api_key":
        if settings.azure_openai_api_key is None:
            raise ConfigurationError("AZURE_OPENAI_API_KEY is required")
        api_key: str | Any = settings.azure_openai_api_key.get_secret_value()
    elif settings.azure_openai_auth_mode == "entra":
        credential_kwargs: dict[str, object] = {}
        if settings.azure_openai_managed_identity_client_id:
            credential_kwargs["managed_identity_client_id"] = (
                settings.azure_openai_managed_identity_client_id
            )
        credential = DefaultAzureCredential(**credential_kwargs)
        api_key = get_bearer_token_provider(credential, settings.azure_openai_token_scope)
    else:
        raise ConfigurationError(
            f"Unsupported AZURE_OPENAI_AUTH_MODE: {settings.azure_openai_auth_mode}"
        )

    return AzureOpenAIClientHandle(
        client=OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(settings.azure_openai_request_timeout_seconds),
            max_retries=settings.azure_openai_max_retries,
        ),
        credential=credential,
    )
