"""Convert raw user photos into the benchmark layout.

    data/UserMainPhotos/*   ->  data/gallery/<person>/clean_01.jpg ...
    data/UserDirtyPhotos/*  ->  data/probes/<person>/dirty/01.jpg ...
    data/UserBadPhotos/*    ->  data/probes/<person>/bad/01.jpg ...
    data/UserFakePhotos/*   ->  data/twins/<person>/01.jpg ...   (look-alikes, not the person)

Every image is decoded (HEIC supported), EXIF orientation is applied, and it
is re-saved as JPEG without metadata (drops GPS/location). Originals are not
modified. A name mapping is written to data/<person>_mapping.csv (stays in
data/, never committed) and an empty label sheet per set to docs/.
Conversion is deterministic, so re-running yields identical files (and
embedding / VLM cache hits). Existing label sheets are never overwritten.

Usage:
    python scripts/prepare_user_data.py --person user01
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from frbench import IMAGE_EXTENSIONS

register_heif_opener()


def convert(src: Path, dst: Path, quality: int = 95) -> tuple[int, int]:
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="JPEG", quality=quality)  # no exif= -> metadata dropped
        return img.size


def images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_labels(path: Path, rows: list[dict]) -> None:
    if path.exists():  # never overwrite labels the user already filled in
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["no", "file", "efekt", "aciklama"])
        for r in rows:
            writer.writerow([r["no"], Path(r["new"]).name, "", ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--person", default="user01")
    ap.add_argument("--main", default="UserMainPhotos")
    ap.add_argument("--dirty", default="UserDirtyPhotos")
    ap.add_argument("--bad", default="UserBadPhotos")
    ap.add_argument("--fake", default="UserFakePhotos")
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    args = ap.parse_args()

    # (set name, source folder, destination folder, file prefix, label sheet)
    sets = [
        ("clean", args.main, args.data / "gallery" / args.person, "clean_", None),
        ("dirty", args.dirty, args.data / "probes" / args.person / "dirty", "", args.docs / "dirty_labels.csv"),
        ("bad", args.bad, args.data / "probes" / args.person / "bad", "", args.docs / "bad_labels.csv"),
        ("twin", args.fake, args.data / "twins" / args.person, "", args.docs / "twin_labels.csv"),
    ]
    rows: list[dict] = []
    for name, folder, dest, prefix, labels in sets:
        src_dir = args.data / folder
        if not src_dir.is_dir():
            print(f"{name}: {src_dir} not found, skipped")
            continue
        set_rows = []
        for i, src in enumerate(images(src_dir), 1):
            dst = dest / f"{prefix}{i:02d}.jpg"
            w, h = convert(src, dst)
            set_rows.append({"set": name, "no": i, "new": str(dst.relative_to(args.data)),
                             "original": src.name, "size": f"{w}x{h}"})
        rows += set_rows
        if labels is not None:
            write_labels(labels, set_rows)
        print(f"{name}: {len(set_rows)} -> {dest}" + (f" (labels: {labels})" if labels else ""))

    mapping = args.data / f"{args.person}_mapping.csv"
    with open(mapping, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"mapping: {mapping}")


if __name__ == "__main__":
    main()
