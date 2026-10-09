"""face-robustness-bench: embedding-based face recognition robustness benchmark."""

__version__ = "0.1.0"

CATEGORIES: tuple[str, ...] = ("filter", "makeup", "effect", "botox", "lens", "injury", "dirty")
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
)
