"""Data contracts and Pydantic models for L.O.R.A Gateway."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Payload for submitting a query to L.O.R.A."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., min_length=1, description="User query or command text")
    notebook_id: str | None = Field(None, description="Optional existing notebook id to append to")
    depth: str | None = Field("standard", description="Requested depth mode (eli5, simple, standard, detailed, expert)")
    run_mode: str = Field("quick", description="Execution mode (quick, standard, deep, deep_with_refinement)")


class AskResponse(BaseModel):
    """Acknowledgement of submitted job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    queue_position: int
    status: str = "queued"


class JobResponse(BaseModel):
    """Status and payload of an asynchronous job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str
    queue_position: int
    origin: str
    result: Any | None = None
    error: str | None = None


class StopRequest(BaseModel):
    """Payload to cancel an ongoing or queued job."""

    model_config = ConfigDict(frozen=True)

    job_id: str


class StopResponse(BaseModel):
    """Response after attempting job cancellation."""

    model_config = ConfigDict(frozen=True)

    stopped: bool
    job_id: str
    message: str


class RewriteRequest(BaseModel):
    """Payload to request re-writing an existing notebook with a different depth."""

    model_config = ConfigDict(frozen=True)

    depth: str = Field("standard", description="Target depth mode")


class NotebookSummary(BaseModel):
    """Summary of a saved research notebook."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    depth_mode: str
    source_count: int
    visibility: str
    created_at: str


class NotebookListResponse(BaseModel):
    """List of notebooks accessible to the caller."""

    model_config = ConfigDict(frozen=True)

    notebooks: list[NotebookSummary]
