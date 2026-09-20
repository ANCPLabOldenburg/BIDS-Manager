"""Run one brainchop model and write its mask. A process of its own.

Not an import away from the rest: this module is executed as
``python -m bidsmgr.deface._mindgrab`` and nothing in the application imports
brainchop or tinygrad at all. That is the point.

**Why a subprocess.** brainchop runs on tinygrad, which sets up a GPU runtime
(Metal on macOS, and the platform equivalent elsewhere) the first time it is
asked to compute. Doing that on a ``QThread`` and then letting the thread go
crashed the application *after* the work finished and the file was written: the
skull strip succeeded, the output was on disk, and the window vanished. A user
reported it exactly that way.

Running it in a separate process removes the whole class of problem rather
than one instance of it. The child sets up whatever runtime it likes, writes a
file, and exits; the parent never loads tinygrad, so nothing it created can
outlive a thread or collide with Qt's own graphics context. It is also how
this package already treats niimath, so there is one story for both engines
instead of two.

The cost is a process start and a fresh import per image, which is a fraction
of a second against ten seconds of inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-mindgrab",
        description="Internal. Run a brainchop model and write its mask.",
    )
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--model", default="mindgrab")
    args = parser.parse_args(argv)

    try:
        import brainchop
    except Exception as exc:  # noqa: BLE001 - reported to the parent verbatim
        print(f"brainchop could not be imported: {exc}", file=sys.stderr)
        return 3

    try:
        volume = brainchop.load(args.source)
        result = brainchop.segment(volume, args.model)
        # brainchop's own writer, because its in-memory array is indexed in
        # the REVERSE axis order to the header it ships with and `save` is
        # what transposes. Pairing the two directly produced a mask rotated
        # with respect to the image, which cut the brain at an angle.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        brainchop.save(result, args.output)
    except Exception as exc:  # noqa: BLE001 - tinygrad raises its own zoo
        print(f"{args.model} failed: {exc}", file=sys.stderr)
        return 1

    if not Path(args.output).is_file():
        print(f"{args.model} reported success but wrote nothing",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
