"""Run brainchop models on one image and write each result. A process of its
own: ``python -m bidsmgr.qc._models SOURCE OUTDIR --models mindgrab robust_tissue``.

Why a subprocess: brainchop runs on tinygrad, which sets up a GPU runtime the
first time it computes, and doing that on a ``QThread`` crashed the
application after the work was done (``deface/_mindgrab.py`` records it). The
parent never imports brainchop or tinygrad. Several models in ONE process,
because loading the image (brainchop conforms it with niimath) and starting
tinygrad cost more than a fast model's own run.

Each result is written with brainchop's own writer, on its conformed 256^3
1 mm grid; ``qc.tools`` brings it back onto the image's grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bidsmgr-qc-models",
                                     description="Internal. Run brainchop models.")
    parser.add_argument("source")
    parser.add_argument("outdir")
    parser.add_argument("--models", nargs="+", default=["mindgrab"])
    args = parser.parse_args(argv)
    try:
        import brainchop
    except Exception as exc:  # noqa: BLE001 - reported to the parent verbatim
        print(f"brainchop could not be imported: {exc}", file=sys.stderr)
        return 3
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        volume = brainchop.load(args.source)
    except Exception as exc:  # noqa: BLE001
        print(f"could not load {args.source}: {exc}", file=sys.stderr)
        return 1
    failed = 0
    for model in args.models:
        try:
            result = brainchop.segment(volume, model)
            brainchop.save(result, str(out / f"{model}.nii.gz"))
        except Exception as exc:  # noqa: BLE001 - tinygrad raises its own zoo
            print(f"{model} failed: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed == len(args.models) else 0


if __name__ == "__main__":
    raise SystemExit(main())
