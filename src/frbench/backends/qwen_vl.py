"""Qwen2.5-VL backend via Ollama: pairwise "same person?" questions.

For every probe, ``gallery_k`` (default 3) reference photos are picked from
the person's gallery (seeded by the probe hash, so the choice is
deterministic but varies across probes). Each (reference, probe) pair is one
query:

    Do these two photos show the same person? Ignore makeup, filters, effects
    and cosmetic changes. Answer JSON: {same: true/false, confidence: 0-100}.

Score per query is the probability-like value
``p_same = confidence/100 if same else min(c, 1 - c)/100`` (see
``Answer.p_same``): a "no" answer never scores above 0.5, whether the model
reports confidence as certainty or as match probability. The probe score is the mean
``p_same`` over valid answers, ``vote_same`` is the majority vote, and the
plain mean confidence is kept as ``mean_confidence``.

Refusals and unparseable answers are failures (never crashes). A probe needs
at least ``min_valid`` valid answers, otherwise it becomes a failure row.
All responses are cached on disk per (model, prompt, reference hash, probe
hash), with temperature 0 and a fixed seed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image, ImageOps

from ..gallery import InsufficientDataError, cached_file_hash
from .base import (
    BACKEND_ERROR,
    INVALID_JSON,
    REFUSAL,
    UNREADABLE,
    ScoreOutcome,
    Template,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_URL = "http://localhost:11434"
PROMPT = (
    "Do these two photos show the same person? Ignore makeup, filters, effects "
    "and cosmetic changes. Answer JSON: {same: true/false, confidence: 0-100}."
)
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "same": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["same", "confidence"],
}
REFUSAL_RE = re.compile(
    r"\b(i\s*(?:can(?:no|')?t|cannot|am unable|'m unable|am not able|won't|will not)"
    r"|unable to (?:help|assist|identify|determine|compare|verify)"
    r"|not able to (?:help|assist|identify|determine|compare|verify)"
    r"|sorry|as an ai|facial recognition is not)\b",
    re.IGNORECASE,
)
FAILURE_PRIORITY = (REFUSAL, INVALID_JSON, BACKEND_ERROR, UNREADABLE)


class OllamaError(RuntimeError):
    """Transport-level failure talking to Ollama."""


class OllamaClient(Protocol):
    def chat(
        self, model: str, prompt: str, images: list[str], options: dict[str, Any], format: Any | None
    ) -> str:
        """Return the assistant message text for one prompt with base64 images."""
        ...

    def check_model(self, model: str) -> None:
        """Raise ``OllamaError`` if the server is down or the model is missing."""
        ...


class HttpOllamaClient:
    """Minimal Ollama REST client (stdlib only)."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600.0, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
        raise OllamaError(f"Ollama request {path} failed: {last}")

    def check_model(self, model: str) -> None:
        names = {m.get("name") for m in self._request("/api/tags").get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            raise OllamaError(
                f"Model '{model}' is not available in Ollama at {self.base_url}. "
                f"Run: ollama pull {model}"
            )

    def chat(self, model, prompt, images, options, format=None) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "stream": False,
            "options": options,
            "keep_alive": "30m",
        }
        if format is not None:
            payload["format"] = format
        return self._request("/api/chat", payload).get("message", {}).get("content", "")


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Answer:
    same: bool | None
    confidence: float | None
    error: str | None = None
    raw: str = ""

    @property
    def valid(self) -> bool:
        return self.error is None

    @property
    def p_same(self) -> float:
        """Probability-like "same person" score in [0, 1].

        "confidence" is ambiguous: models use it either as certainty in the
        answer (``same: false, confidence: 95`` = surely different) or as the
        probability of a match (``same: false, confidence: 0``, observed with
        qwen2.5vl). Mapping a "different" answer to ``min(c, 1 - c)`` is
        correct under both readings and keeps it at or below 0.5.
        """
        assert self.same is not None and self.confidence is not None
        c = self.confidence / 100.0
        return c if self.same else min(c, 1.0 - c)


def _extract_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "false", "no"}:
        return value.strip().lower() in {"true", "yes"}
    return None


def _as_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return conf if 0.0 <= conf <= 100.0 else None


def parse_answer(text: str) -> Answer:
    """Parse the model reply; classify failures as refusal or invalid_json."""
    try:
        obj = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        kind = REFUSAL if REFUSAL_RE.search(text or "") else INVALID_JSON
        return Answer(None, None, kind, text)
    if not isinstance(obj, dict):
        return Answer(None, None, INVALID_JSON, text)
    same = _as_bool(obj.get("same"))
    conf = _as_confidence(obj.get("confidence"))
    if same is None or conf is None:
        return Answer(None, None, INVALID_JSON, text)
    return Answer(same, conf, None, text)


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


class ResponseCache:
    """One JSON file per (model, prompt, options, reference hash, probe hash)."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        if self.directory is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())["response"]

    def put(self, key: str, record: dict[str, Any]) -> None:
        if self.directory is None:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(path)


# --------------------------------------------------------------------------- #
# Recognizer
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=512)
def encode_image(path: str, max_side: int) -> str:
    """EXIF-corrected, RGB, downscaled JPEG as base64 (Ollama input format)."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class QwenVLRecognizer:
    name = "qwen_vl"

    def __init__(
        self,
        client: OllamaClient | None = None,
        model: str = DEFAULT_MODEL,
        gallery_k: int = 3,
        min_valid: int = 2,
        seed: int = 42,
        max_side: int = 768,
        cache_dir: Path | None = None,
        max_impostors_per_person: int | None = 60,
        structured: bool = False,
    ) -> None:
        self.client = client or HttpOllamaClient()
        self.model = model
        self.model_name = f"ollama:{model}"
        self.gallery_k = gallery_k
        self.min_valid = min_valid
        self.seed = seed
        self.max_side = max_side
        self.max_impostors_per_person = max_impostors_per_person
        # structured=True makes Ollama enforce the JSON schema. Off by default
        # so that refusals stay observable, as the benchmark wants to count them.
        self.format = RESPONSE_SCHEMA if structured else None
        self.options = {"temperature": 0, "seed": seed}
        self.cache = ResponseCache(cache_dir / "vlm" / model.replace(":", "_").replace("/", "_") if cache_dir else None)
        self.stats: Counter[str] = Counter()
        self._checked = False
        self._config_hash = hashlib.sha256(
            json.dumps(
                {"model": model, "prompt": PROMPT, "options": self.options,
                 "format": self.format, "max_side": max_side},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

    # -- template ----------------------------------------------------------- #

    def preflight(self) -> None:
        """Fail fast if Ollama is down or the model is not pulled."""
        if not self._checked:
            self.client.check_model(self.model)
            self._checked = True

    def enroll(self, person_id: str, gallery: Sequence[Path]) -> Template:
        if not gallery:
            raise InsufficientDataError("No gallery photos to compare against.")
        self.preflight()
        return Template(person_id, sorted(gallery))

    def references(self, gallery: Sequence[Path], probe_sha: str) -> list[Path]:
        """Deterministic, probe-specific choice of ``gallery_k`` reference photos."""
        if len(gallery) <= self.gallery_k:
            return list(gallery)
        return random.Random(f"{self.seed}:{probe_sha}").sample(list(gallery), self.gallery_k)

    # -- querying ----------------------------------------------------------- #

    def cache_key(self, ref_sha: str, probe_sha: str) -> str:
        return hashlib.sha256(f"{self._config_hash}|{ref_sha}|{probe_sha}".encode()).hexdigest()

    def ask(self, ref: Path, probe: Path) -> Answer:
        ref_sha, probe_sha = cached_file_hash(ref), cached_file_hash(probe)
        key = self.cache_key(ref_sha, probe_sha)
        self.stats["queries"] += 1
        text = self.cache.get(key)
        if text is not None:
            self.stats["cache_hits"] += 1
        else:
            try:
                images = [encode_image(str(ref), self.max_side), encode_image(str(probe), self.max_side)]
            except Exception as exc:
                self.stats[UNREADABLE] += 1
                return Answer(None, None, UNREADABLE, f"{type(exc).__name__}: {exc}")
            try:
                text = self.client.chat(self.model, PROMPT, images, dict(self.options), self.format)
            except Exception as exc:  # transport errors are not cached: retry next run
                self.stats[BACKEND_ERROR] += 1
                return Answer(None, None, BACKEND_ERROR, f"{type(exc).__name__}: {exc}")
            self.cache.put(key, {
                "model": self.model, "reference_sha256": ref_sha, "probe_sha256": probe_sha,
                "config": self._config_hash, "response": text,
            })
            if self.stats["queries"] % 25 == 0:
                log.info("qwen_vl: %d queries (%d from cache)", self.stats["queries"], self.stats["cache_hits"])
        answer = parse_answer(text)
        if answer.error:
            self.stats[answer.error] += 1
        return answer

    def aggregate(self, probe: Path, answers: list[Answer]) -> ScoreOutcome:
        valid = [a for a in answers if a.valid]
        extra: dict[str, Any] = {"n_queries": len(answers), "n_valid": len(valid)}
        if len(valid) < min(self.min_valid, len(answers)) or not valid:
            counts = Counter(a.error for a in answers if a.error)
            kind = max(counts, key=lambda k: (counts[k], -FAILURE_PRIORITY.index(k)))
            msg = "; ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            return ScoreOutcome(probe, None, kind, msg, extra)
        p = [a.p_same for a in valid]
        n_same = sum(bool(a.same) for a in valid)
        score = sum(p) / len(p)
        extra.update(
            vote_same=n_same * 2 > len(valid) or (n_same * 2 == len(valid) and score >= 0.5),
            n_same=n_same,
            mean_confidence=sum(a.confidence for a in valid) / len(valid),
        )
        return ScoreOutcome(probe, score, None, "", extra)

    def score(self, template: Template, paths: Sequence[Path]) -> list[ScoreOutcome]:
        out = []
        for probe in paths:
            try:
                probe_sha = cached_file_hash(probe)
            except OSError as exc:
                out.append(ScoreOutcome(probe, None, UNREADABLE, str(exc)))
                continue
            refs = self.references(template.data, probe_sha)
            out.append(self.aggregate(probe, [self.ask(r, probe) for r in refs]))
        return out

    def finish(self) -> dict[str, Any]:
        log.info("qwen_vl stats: %s", dict(self.stats))
        return dict(self.stats)
