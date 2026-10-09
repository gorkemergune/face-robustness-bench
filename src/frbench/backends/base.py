"""Recognizer interface shared by all backends.

A recognizer enrolls a person from their gallery photos into a ``Template``
and then scores probe photos against it. Higher score = more likely the same
person. The benchmark calibrates tau on these scores exactly the same way for
every backend, so their recognition rates are comparable at the same FAR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

# Failure kinds (``ScoreOutcome.error``).
NO_FACE = "no_face"
UNREADABLE = "unreadable"
REFUSAL = "refusal"
INVALID_JSON = "invalid_json"
BACKEND_ERROR = "backend_error"
FAILURE_KINDS: tuple[str, ...] = (NO_FACE, UNREADABLE, REFUSAL, INVALID_JSON, BACKEND_ERROR)


@dataclass(frozen=True)
class ScoreOutcome:
    """Score of one image against one template; ``score is None`` on failure."""

    path: Path
    score: float | None
    error: str | None = None  # one of FAILURE_KINDS
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.score is not None


@dataclass
class Template:
    person_id: str
    data: Any
    gallery_outcomes: list[ScoreOutcome] = field(default_factory=list)


@runtime_checkable
class Recognizer(Protocol):
    name: str  # short backend id, e.g. "arcface", "qwen_vl"
    model_name: str  # e.g. "insightface:buffalo_l", "ollama:qwen2.5vl:7b"
    # Cap on impostor images per template (None = all). Expensive backends
    # use a seeded subsample so a run stays tractable.
    max_impostors_per_person: int | None

    def enroll(self, person_id: str, gallery: Sequence[Path]) -> Template:
        """Build a template; raise ``InsufficientDataError`` if impossible."""
        ...

    def score(self, template: Template, paths: Sequence[Path]) -> list[ScoreOutcome]:
        """Score each image against the template, never raising per image."""
        ...

    def finish(self) -> dict[str, Any]:
        """Persist caches; return run statistics for the report."""
        ...
