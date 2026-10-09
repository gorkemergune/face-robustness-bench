"""Recognition backends: ArcFace embeddings and Qwen2.5-VL (Ollama)."""

from .arcface import ArcFaceRecognizer
from .base import FAILURE_KINDS, Recognizer, ScoreOutcome, Template
from .qwen_vl import QwenVLRecognizer

BACKENDS: tuple[str, ...] = ("arcface", "qwen_vl")

__all__ = [
    "ArcFaceRecognizer",
    "BACKENDS",
    "FAILURE_KINDS",
    "QwenVLRecognizer",
    "Recognizer",
    "ScoreOutcome",
    "Template",
]
