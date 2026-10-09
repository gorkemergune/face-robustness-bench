"""ArcFace backend: embedding centroid template + cosine similarity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..embedder import EmbedOutcome, Embedder, safe_embed
from ..gallery import InsufficientDataError, build_centroid, cached_file_hash
from ..scoring import cosine_many
from .base import BACKEND_ERROR, NO_FACE, UNREADABLE, ScoreOutcome, Template

log = logging.getLogger(__name__)

MIN_GALLERY_EMBEDDINGS = 3


class EmbeddingStore:
    """Embeds each distinct file once; caches results by SHA-256 + model name."""

    def __init__(self, embedder: Embedder, cache_path: Path | None = None) -> None:
        self.embedder = embedder
        self.cache_path = cache_path
        self.computed = 0
        self._mem: dict[str, EmbedOutcome] = {}
        self._disk: dict[str, np.ndarray] = {}
        if cache_path and cache_path.exists():
            with np.load(cache_path) as npz:
                self._disk = {k: npz[k] for k in npz.files}
            log.info("Loaded %d cached embeddings from %s", len(self._disk), cache_path)

    def get(self, path: Path) -> EmbedOutcome:
        digest = cached_file_hash(path)
        if digest in self._mem:
            cached = self._mem[digest]
            return EmbedOutcome(path, cached.embedding, cached.error)
        if digest in self._disk:
            arr = self._disk[digest]
            outcome = EmbedOutcome(path, arr if arr.size else None, None if arr.size else NO_FACE)
        else:
            self.computed += 1
            outcome = safe_embed(self.embedder, path)
            if outcome.embedding is not None:
                self._disk[digest] = outcome.embedding
            elif outcome.error == NO_FACE:
                self._disk[digest] = np.empty(0, dtype=np.float32)
        self._mem[digest] = outcome
        return outcome

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, **self._disk)
        tmp.replace(self.cache_path)


def _error_kind(error: str | None) -> str | None:
    """Map ``safe_embed`` messages ('no_face', 'unreadable: ...', 'error: ...') to kinds."""
    if error is None:
        return None
    head = error.split(":", 1)[0]
    return {NO_FACE: NO_FACE, UNREADABLE: UNREADABLE}.get(head, BACKEND_ERROR)


class ArcFaceRecognizer:
    name = "arcface"
    max_impostors_per_person: int | None = None

    def __init__(self, embedder: Embedder, cache_dir: Path | None = None) -> None:
        self.model_name = embedder.name
        cache_path = None
        if cache_dir is not None:
            safe = embedder.name.replace(":", "_").replace("/", "_")
            cache_path = cache_dir / f"embeddings_{safe}.npz"
        self.store = EmbeddingStore(embedder, cache_path)

    def _to_scores(self, outcomes: list[EmbedOutcome], centroid: np.ndarray) -> list[ScoreOutcome]:
        idx = [i for i, o in enumerate(outcomes) if o.embedding is not None]
        cos = np.full(len(outcomes), np.nan)
        if idx:
            cos[idx] = cosine_many(np.stack([outcomes[i].embedding for i in idx]), centroid)
        return [
            ScoreOutcome(o.path, None if np.isnan(c) else float(c), _error_kind(o.error), o.error or "")
            for o, c in zip(outcomes, cos)
        ]

    def enroll(self, person_id: str, gallery: Sequence[Path]) -> Template:
        outcomes = [self.store.get(p) for p in gallery]
        embs = [o.embedding for o in outcomes if o.embedding is not None]
        if len(embs) < MIN_GALLERY_EMBEDDINGS:
            raise InsufficientDataError(
                f"Only {len(embs)} of {len(outcomes)} gallery photos had a detectable face "
                f"(need {MIN_GALLERY_EMBEDDINGS})."
            )
        centroid = build_centroid(embs)
        return Template(person_id, centroid, self._to_scores(outcomes, centroid))

    def score(self, template: Template, paths: Sequence[Path]) -> list[ScoreOutcome]:
        return self._to_scores([self.store.get(p) for p in paths], template.data)

    def audit(self, template: Template, paths: Sequence[Path]) -> list[dict[str, Any]]:
        """Face count and best-matching face per image (multi-person photos).

        The score always uses the largest face; this shows whether another face
        in the photo matches the person better (i.e. the wrong face was picked).
        Skipped when the embedder cannot return all faces.
        """
        embed_all = getattr(self.store.embedder, "embed_all", None)
        if not callable(embed_all):
            return [{} for _ in paths]
        out: list[dict[str, Any]] = []
        for path in paths:
            try:
                faces = embed_all(path)
            except Exception as exc:  # audit is informational only
                log.warning("Face audit failed for %s: %s", path, exc)
                out.append({})
                continue
            best = max((float(e @ template.data) for _, e in faces), default=float("nan"))
            out.append({"n_faces": len(faces), "best_face_score": best})
        return out

    def finish(self) -> dict[str, Any]:
        self.store.save()
        return {"embeddings_computed": self.store.computed}
