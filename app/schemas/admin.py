"""Contracts for the disabled-by-default scoped demo reset."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResetDemoRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "confirmation": "RESET_PROJECT_AURORA_DEMO",
                    "dry_run": True,
                }
            ]
        },
    )

    confirmation: Literal["RESET_PROJECT_AURORA_DEMO"]
    dry_run: bool = True


class ResetDemoResponse(BaseModel):
    dry_run: bool
    executed: bool
    reset_enabled: bool
    resources: dict[str, int]
    chroma_collections: list[str] = Field(default_factory=list)
    managed_files: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
