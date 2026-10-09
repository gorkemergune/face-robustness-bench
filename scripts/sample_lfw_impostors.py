"""Copy a seeded sample of LFW identities into data/impostors/ for tau calibration.

Download and extract LFW first (e.g. lfw-funneled.tgz from
https://ndownloader.figshare.com/files/5976015), then:

    python scripts/sample_lfw_impostors.py --lfw /path/to/lfw_funneled --n 1000
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lfw", type=Path, required=True, help="extracted lfw_funneled/ folder")
    ap.add_argument("--out", type=Path, default=Path("data/impostors"))
    ap.add_argument("--n", type=int, default=1000, help="number of identities")
    ap.add_argument("--per-person", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    people = sorted(p for p in args.lfw.iterdir() if p.is_dir())
    chosen = sorted(random.Random(args.seed).sample(people, min(args.n, len(people))))
    copied = 0
    for person in chosen:
        for img in sorted(person.glob("*.jpg"))[: args.per_person]:
            dst = args.out / f"lfw_{person.name}" / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(img, dst)
            copied += 1
    print(f"Copied {copied} images of {len(chosen)} LFW identities to {args.out}")


if __name__ == "__main__":
    main()
