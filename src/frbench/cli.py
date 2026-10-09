"""Command line interface.

    python -m frbench.cli check --data data/
    python -m frbench.cli run   --data data/ --far 0.01
    python -m frbench.cli run   --data data/ --backends arcface,qwen_vl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import CATEGORIES
from .backends import BACKENDS, ArcFaceRecognizer, QwenVLRecognizer, Recognizer
from .backends.qwen_vl import DEFAULT_MODEL, DEFAULT_URL, HttpOllamaClient, OllamaError
from .embedder import InsightFaceEmbedder
from .effects import write_effects_report
from .evaluate import BenchConfig, run_all, save_raw, subdirs
from .gallery import MIN_CLEAN_PHOTOS, dedupe_by_hash, list_images
from .report import comparison_table, label, markdown_table, summarize, write_reports

log = logging.getLogger("frbench")

# factory(args, cache_dir) -> Recognizer
RecognizerFactory = Callable[[argparse.Namespace, Path | None], Recognizer]


def make_arcface(args: argparse.Namespace, cache_dir: Path | None) -> Recognizer:
    embedder = InsightFaceEmbedder(model_name=args.model, det_size=(args.det_size, args.det_size))
    return ArcFaceRecognizer(embedder, cache_dir)


def make_qwen(args: argparse.Namespace, cache_dir: Path | None) -> Recognizer:
    return QwenVLRecognizer(
        client=HttpOllamaClient(args.ollama_url),
        model=args.qwen_model,
        gallery_k=args.qwen_gallery_k,
        min_valid=args.qwen_min_valid,
        seed=args.seed,
        max_side=args.qwen_max_side,
        cache_dir=cache_dir,
        max_impostors_per_person=args.qwen_max_impostors or None,
        structured=args.qwen_structured,
    )


DEFAULT_FACTORIES: dict[str, RecognizerFactory] = {"arcface": make_arcface, "qwen_vl": make_qwen}
ALIASES = {"qwen": "qwen_vl", "qwen-vl": "qwen_vl", "arc": "arcface"}


def parse_backends(value: str) -> list[str]:
    names = [ALIASES.get(v.strip().lower(), v.strip().lower()) for v in value.split(",") if v.strip()]
    unknown = [n for n in names if n not in BACKENDS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown backend(s) {unknown}; choose from {list(BACKENDS)}")
    return list(dict.fromkeys(names))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frbench", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the benchmark")
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--results-md", default="docs/RESULTS.md",
                     help="where to write the results document ('' to skip)")
    run.add_argument("--backends", type=parse_backends, default=["arcface"],
                     help=f"comma-separated: {','.join(BACKENDS)} (default: arcface)")
    run.add_argument("--far", type=float, default=0.01, help="target false accept rate")
    run.add_argument("--tau", type=float, default=None, help="fixed threshold (skip calibration)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--gallery-fraction", type=float, default=0.6)
    run.add_argument("--min-clean", type=int, default=MIN_CLEAN_PHOTOS)
    run.add_argument("--no-cross-impostors", action="store_true",
                     help="use only data/impostors, not other enrolled persons")
    run.add_argument("--no-cache", action="store_true", help="do not cache embeddings / VLM answers")

    arc = run.add_argument_group("arcface")
    arc.add_argument("--model", default="buffalo_l")
    arc.add_argument("--det-size", type=int, default=640)

    qwen = run.add_argument_group("qwen_vl (Ollama)")
    qwen.add_argument("--ollama-url", default=DEFAULT_URL)
    qwen.add_argument("--qwen-model", default=DEFAULT_MODEL)
    qwen.add_argument("--qwen-gallery-k", type=int, default=3, help="gallery photos per probe")
    qwen.add_argument("--qwen-min-valid", type=int, default=2,
                      help="valid answers needed per probe, else failure row")
    qwen.add_argument("--qwen-max-side", type=int, default=768, help="resize images before sending")
    qwen.add_argument("--qwen-max-impostors", type=int, default=60,
                      help="seeded impostor subsample per person (0 = all)")
    qwen.add_argument("--qwen-structured", action="store_true",
                      help="let Ollama enforce the JSON schema (hides refusals)")

    eff = sub.add_parser("effects", help="per-photo / per-effect report (Turkish) from run outputs")
    eff.add_argument("--results", type=Path, default=Path("results"))
    eff.add_argument("--labels", type=Path, default=None,
                     help="label sheet for --category (default: docs/<category>_labels.csv)")
    eff.add_argument("--category", default=None,
                     help="only this category (default: all, e.g. dirty, bad, twin)")
    eff.add_argument("--out", type=Path, default=Path("docs/RAPOR.md"))

    check = sub.add_parser("check", help="validate the data layout (no model needed)")
    check.add_argument("--data", type=Path, default=Path("data"))
    check.add_argument("--min-clean", type=int, default=MIN_CLEAN_PHOTOS)
    return parser


def cmd_check(args: argparse.Namespace) -> int:
    data: Path = args.data
    ok = True
    people = subdirs(data / "gallery")
    if not people:
        print(f"ERROR: no person folders in {data / 'gallery'}")
        return 1
    print(f"{'person':<20} {'clean':>5} " + " ".join(f"{c:>7}" for c in CATEGORIES) + "  other")
    for person in people:
        n_clean = len(dedupe_by_hash(list_images(person)))
        probe_dir = data / "probes" / person.name
        counts = {c.name.lower(): len(list_images(c)) for c in subdirs(probe_dir)}
        other = {k: v for k, v in counts.items() if k not in CATEGORIES}
        flag = "" if n_clean >= args.min_clean else f"  <-- needs >= {args.min_clean} clean"
        ok &= n_clean >= args.min_clean
        print(
            f"{person.name:<20} {n_clean:>5} "
            + " ".join(f"{counts.get(c, 0):>7}" for c in CATEGORIES)
            + (f"  {other}" if other else "")
            + flag
        )
    imp = subdirs(data / "impostors")
    n_imp = sum(len(list_images(d)) for d in imp) + len(list_images(data / "impostors"))
    print(f"\nimpostor ids: {len(imp)}, impostor photos: {n_imp}")
    print(
        "cross-person impostor pairs available from enrolled persons: "
        f"{max(0, len(people) - 1)} other person(s) per template"
    )
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace, factories: Mapping[str, RecognizerFactory]) -> int:
    cfg = BenchConfig(
        data_dir=args.data,
        out_dir=args.out,
        far=args.far,
        seed=args.seed,
        gallery_fraction=args.gallery_fraction,
        min_clean=args.min_clean,
        manual_tau=args.tau,
        cross_impostors=not args.no_cross_impostors,
        use_cache=not args.no_cache,
    )
    cache_dir = None if args.no_cache else cfg.out_dir / "cache"
    recognizers = [factories[name](args, cache_dir) for name in args.backends]
    results = run_all(cfg, recognizers)
    raw = save_raw(results)
    results_md = Path(args.results_md) if args.results_md else None
    paths = write_reports(results, results_md)

    for r in results:
        t = r.threshold
        print(f"\n{label(r.backend)} ({r.model_name}) | tau={t.tau:.4f} | FAR={t.far_effective:.2%} "
              f"| persons={len(r.persons)}\n")
        print(markdown_table(summarize(r.rows, r.categories), r.backend))
    if len(results) > 1:
        print("\nComparison\n")
        print(comparison_table(results))
    for w in dict.fromkeys(w for r in results for w in r.warnings):
        print(f"WARNING: {w}")
    print(f"\nWrote {len(raw)} raw CSV(s) to {cfg.out_dir / 'raw'}")
    for name, p in paths.items():
        print(f"Wrote {name}: {p}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    factories: Mapping[str, RecognizerFactory] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "check":
        return cmd_check(args)
    if args.command == "effects":
        print(f"Wrote {write_effects_report(args.results, args.labels, args.out, args.category)}")
        return 0
    try:
        return cmd_run(args, {**DEFAULT_FACTORIES, **(factories or {})})
    except OllamaError as exc:
        print(f"ERROR: {exc}\nIs Ollama running? Start it with `ollama serve` "
              f"and pull the model with `ollama pull {args.qwen_model}`.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
