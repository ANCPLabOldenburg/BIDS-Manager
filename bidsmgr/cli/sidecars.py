"""``bidsmgr-sidecars``: write the sidecars recordings are missing.

The Editor's "Sidecars from the recordings" without the window, over the same
engine (:mod:`bidsmgr.editor.sidecars`). A sidecar is written from the data a
recording was converted from (in a project that converted it), its own header,
the answers given in the project, the sidecars that link to it and its name,
never over a value already there, in the format of the chosen BIDS version.
Any image or recording BIDS Manager reads gets one, named the BIDS way or not:
its sidecar is written beside it.

Plans first and prints the plan (``--dry-run`` stops there); written, the
change is one step of the Editor's history, undoable from the Editor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..editor import sidecars as engine
from ..metadata.engine import FILL_SCOPES
from ..util.oplog import add_log_arguments, dataset_root_for, log_keywords, logged

_KIND_WORDS = {
    engine.KIND_BIDS: "BIDS name",
    engine.KIND_NAME: "from its name",
    engine.KIND_FOLDER: "from its folder",
    engine.KIND_GUESS: "guessed",
    engine.KIND_CHOSEN: "as given",
}


def _kind_arg(text: str) -> tuple[str, tuple[str, str]]:
    """``PATH=DATATYPE/SUFFIX`` -> ``(PATH, (DATATYPE, SUFFIX))``."""
    path, eq, kind = text.rpartition("=")
    datatype, slash, suffix = kind.partition("/")
    if not (eq and path and slash and datatype and suffix):
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected PATH=DATATYPE/SUFFIX, for example "
            "mri/t1_mprage.nii=anat/T1w")
    return path, (datatype, suffix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-sidecars",
        description=(
            "Write the .json sidecar a recording is missing, or add what one lacks "
            "(--complete), from the data it was converted from, its own header "
            "(NIfTI timing, the spectroscopy header, what MNE-BIDS reads from an EEG "
            "or MEG recording), the project's answers, the sidecars that link to it "
            "and its name. A value already in a sidecar is never changed, and a value "
            "the chosen BIDS version does not allow is never written. Any image or "
            "recording BIDS Manager reads gets one, named the BIDS way or not."),
    )
    parser.add_argument("root", type=Path,
                        help="The dataset, or any folder holding recordings.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Files or folders to do, inside ROOT (default: the whole "
                             "dataset, less derivatives/, sourcedata/ and what "
                             ".bidsignore hides; those are done when named here).")
    parser.add_argument("--complete", action="store_true",
                        help="Also add to sidecars that are there: they keep every "
                             "value and gain what they lack (a TODO is replaced by a "
                             "value that was found).")
    parser.add_argument("--no-source", action="store_true",
                        help="Do not convert the source data again, even in a project "
                             "that converted it: read the files alone (faster).")
    parser.add_argument("--fill-scope", choices=FILL_SCOPES, default="none",
                        help="Mark what is still unknown with TODO, as the metadata step "
                             "does: required, recommended (required and recommended), "
                             "optional (every declared field), or none (default).")
    parser.add_argument("--kind", type=_kind_arg, action="append", default=[],
                        metavar="PATH=DATATYPE/SUFFIX",
                        help="What a file not named the BIDS way is recorded as, when "
                             "the guess is wrong (repeatable). PATH is relative to ROOT.")
    parser.add_argument("--schema", default=None, metavar="VERSION",
                        help="Which BIDS version the fields and values follow (for "
                             "example 1.11.1). Default: the newest version shipped.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written; write nothing.")
    parser.add_argument("--fields", action="store_true",
                        help="Print every field with where it was read.")
    parser.add_argument("--json", action="store_true",
                        help="Print the plan as JSON (with --dry-run, for a script).")
    add_log_arguments(parser)
    return parser


def _relative(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root
    if not root.is_dir():
        parser.error(f"{root} is not a folder")
    from ..schema import set_active_version

    set_active_version(args.schema)
    targets = [_relative(root, p) for p in args.paths] or None
    for target in targets or []:
        if not target.exists():
            parser.error(f"{target} does not exist")
        try:
            target.resolve().relative_to(root.resolve())
        except ValueError:
            parser.error(f"{target} is not inside {root}")
    kinds = {Path(path).as_posix(): kind for path, kind in args.kind}
    return run_sidecars_cli(
        root, targets, kinds=kinds, complete=args.complete, use_source=not args.no_source,
        fill_scope=args.fill_scope, dry_run=args.dry_run, fields=args.fields,
        as_json=args.json, **log_keywords(args),
    )


@logged("sidecars", lambda a: (
    dataset_root_for(Path(a["root"])),
    {"root": a["root"], "paths": [str(p) for p in a["targets"] or []],
     "complete": a["complete"], "source data": a["use_source"],
     "fill scope": a["fill_scope"], "dry run": a["dry_run"]}))
def run_sidecars_cli(root: Path, targets=None, *, kinds: Optional[dict] = None,
                     complete: bool = False, use_source: bool = True,
                     fill_scope: str = "none", dry_run: bool = False,
                     fields: bool = False, as_json: bool = False) -> int:
    """Plan, print and (unless ``dry_run``) write. Returns the exit code."""
    plan = engine.plan_sidecars(
        root, targets, complete=complete, use_source=use_source,
        fill_scope=fill_scope, kinds=kinds or {},
    )
    if as_json:
        print(json.dumps(_as_json(plan), indent=2, ensure_ascii=False, default=str))
    else:
        _print(plan, fields=fields)
    if dry_run:
        return 0
    written = engine.apply_sidecars(root, plan.writes())
    if not as_json:
        n = len(written)
        print(f"Wrote {n} sidecar{'s' if n != 1 else ''}. The Editor's Undo takes "
              f"{'it' if n == 1 else 'them'} back." if n else "Nothing to write.")
    return 0


def _origins(one) -> str:
    counts = one.by_origin()
    return ", ".join(f"{engine.ORIGINS[o].lower()} {counts[o]}"
                     for o in engine.ORIGINS if o in counts)


def _print(plan, *, fields: bool) -> None:
    writes = plan.writes()
    new = sum(1 for p in writes if not p.exists)
    print(f"{plan.root.name}: {new} new sidecar{'s' if new != 1 else ''}, "
          f"{len(writes) - new} added to, BIDS {plan.bids_version}"
          + (", source data converted again" if plan.used_source else "")
          + (f", anonymised as the dataset was ({plan.anonymised})" if plan.anonymised else ""))
    for one in plan.files:
        kind = one.kind
        how = f"{kind.datatype}/{kind.suffix}, {_KIND_WORDS.get(kind.how, kind.how)}"
        if not one.changes:
            print(f"  nothing to add  {one.sidecar}  [{how}]")
            continue
        verb = "add" if one.exists else "new"
        print(f"  {verb:<14} {one.sidecar}  [{how}]  {len(one.added)} fields: {_origins(one)}")
        if not kind.bids:
            print(f"      note: Not a BIDS name or place: read as {kind.datatype}/"
                  f"{kind.suffix} ({kind.why}; --kind to correct it), and the sidecar is "
                  "written beside it.")
        for note in one.notes:
            print(f"      note: {note}")
        for r in one.rejected:
            print(f"      not written: {r.problem} ({json.dumps(r.value, default=str)[:60]})")
        if one.still_missing:
            print(f"      still missing (required): {', '.join(one.still_missing)}")
        if fields:
            for f in one.added:
                value = json.dumps(f.value, ensure_ascii=False, default=str)
                print(f"      {f.name} = {value[:80]}  [{engine.ORIGINS[f.origin]}]")
    have = [rel for rel, why in plan.skipped if why == "has a sidecar"]
    if have:
        print(f"  {len(have)} recording{'s' if len(have) != 1 else ''} already "
              f"{'has its' if len(have) == 1 else 'have their'} sidecar "
              "(--complete adds what they lack)")
    for rel, why in plan.skipped:
        if why != "has a sidecar":
            print(f"  left alone      {rel}: {why}")


def _as_json(plan) -> dict:
    return {
        "root": str(plan.root),
        "bids_version": plan.bids_version,
        "used_source": plan.used_source,
        "anonymised": plan.anonymised,
        "files": [{
            "recording": one.recording,
            "sidecar": one.sidecar,
            "exists": one.exists,
            "kind": {"datatype": one.kind.datatype, "suffix": one.kind.suffix,
                     "how": one.kind.how, "why": one.kind.why},
            "added": [{"field": f.name, "value": f.value, "origin": f.origin,
                       "replaces_todo": f.replaces_todo, "overrides": f.overrides}
                      for f in one.added],
            "inherited": [{"field": i.name, "value": i.value, "source": i.source,
                           "differs": i.differs} for i in one.inherited],
            "rejected": [{"field": r.name, "value": r.value, "origin": r.origin,
                          "problem": r.problem} for r in one.rejected],
            "still_missing": one.still_missing,
            "read": one.read,
            "notes": one.notes,
        } for one in plan.files],
        "skipped": [{"recording": r, "why": w} for r, w in plan.skipped],
    }


if __name__ == "__main__":
    sys.exit(main())
