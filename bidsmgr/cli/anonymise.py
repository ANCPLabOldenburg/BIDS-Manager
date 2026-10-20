"""``bidsmgr-anonymise``: anonymise a BIDS dataset's metadata.

For any BIDS dataset, converted here or not. Plans every change first and
prints it (``--dry-run`` stops there); applied, the change is one step of the
Editor's history (undoable from the Editor). See ``bidsmgr.anonymise``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..anonymise import CATEGORIES, PRESETS, apply_anonymise, plan_anonymise, policy_for
from ..anonymise.scope import Scope


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-anonymise",
        description="Anonymise a BIDS dataset's metadata: dates shifted per participant "
                    "into 1902-1925 (intervals and weekdays kept), names, IDs and birth "
                    "dates removed, UIDs and devices pseudonymised, ages capped at 89, "
                    "gzip headers and photo metadata cleaned. Recordings' data are never "
                    "changed.")
    parser.add_argument("bids_root", type=Path, help="The dataset (its root folder).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="recommended",
                        help="recommended (default) keeps times of day and the site name; "
                             "strict sets times to midnight, removes the site and bins ages.")
    parser.add_argument("--keep", nargs="+", default=(), metavar="CATEGORY",
                        choices=[c.key for c in CATEGORIES],
                        help="Categories to leave alone: "
                             + "; ".join(f"{c.key} ({c.title.lower()})" for c in CATEGORIES)
                             + ".")
    parser.add_argument("--participant-label", nargs="+", default=None, metavar="LABEL",
                        help="Only these participants (sub- prefix optional).")
    parser.add_argument("--session", nargs="+", default=None, metavar="LABEL",
                        help="Only these sessions of the chosen participants (ses- prefix "
                             "optional). Their dates move as the rest of the participant's "
                             "will, so anonymising the others later keeps the intervals.")
    parser.add_argument("--skip-dataset-files", action="store_true",
                        help="Leave the files at the top (participants.tsv and the like) "
                             "alone.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List every change, the values to review by hand and what "
                             "cannot be changed. Nothing in the dataset is written; the "
                             "date offsets chosen are remembered in .bidsmgr/, so the real "
                             "run makes exactly the changes listed.")
    parser.add_argument("--json", action="store_true",
                        help="Print the plan as JSON (with --dry-run, for a script).")
    args = parser.parse_args(argv)

    root = args.bids_root
    if not (root / "dataset_description.json").is_file():
        parser.error(f"{root} has no dataset_description.json: not a BIDS dataset")
    subjects = None
    if args.participant_label:
        subjects = frozenset(s if s.startswith("sub-") else f"sub-{s}"
                             for s in args.participant_label)
    sessions = None
    if args.session:
        labels = [s if s.startswith("ses-") else f"ses-{s}" for s in args.session]
        chosen = subjects if subjects is not None else frozenset(
            p.name for p in root.glob("sub-*") if p.is_dir())
        sessions = frozenset(f"{sub}/{ses}" for sub in chosen for ses in labels
                             if (root / sub / ses).is_dir())
    scope = Scope(top=not args.skip_dataset_files, subjects=subjects, sessions=sessions)
    policy = policy_for(args.preset, keep=tuple(args.keep))
    plan = plan_anonymise(root, policy, scope=scope)

    if args.json:
        print(json.dumps({
            "preset": args.preset,
            "categories": policy.categories_on(),
            "scope": scope.describe(),
            "files": [{"path": plan.rel(f.path), "changes": [
                {"what": c.what, "before": c.before, "after": c.after, "why": c.reason,
                 "category": c.category}
                for c in f.changes]} for f in plan.files],
            "review": [{"path": plan.rel(p), "what": r.what, "value": r.value,
                        "why": r.reason} for p, r in plan.reviews],
            "cannot": [{"path": p, "why": w} for p, w in plan.cannot],
        }, indent=2, ensure_ascii=False))
    else:
        print(f"{root.name} ({policy.label}, {scope.describe()}): {plan.summary()}")
        for planned in plan.files:
            print(f"  {plan.rel(planned.path)}")
            for change in planned.changes:
                after = change.after or "(removed)"
                print(f"    {change.what}: {change.before} -> {after}  [{change.reason}]")
        if plan.reviews:
            print("To review by hand (not changed):")
            for what, value, why, paths in plan.grouped_reviews():
                where = (plan.rel(paths[0]) if len(paths) == 1
                         else f"{len(paths)} files, e.g. {plan.rel(paths[0])}")
                print(f"  {what} = {value!r} in {where}  [{why}]")
        if plan.cannot:
            print("Not changed:")
            for path, why in plan.cannot:
                print(f"  {path}: {why}")
    if args.dry_run:
        if plan.state_changed:
            # The dataset is not touched; the date offsets just chosen are
            # remembered (in .bidsmgr/), so the real run shifts by what this
            # one showed rather than by new ones.
            from ..anonymise.state import STATE_FILE

            target = root / STATE_FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(plan.state.to_json(), encoding="utf-8")
        return 0

    from ..project.operations import begin_operation

    with begin_operation(root, f"Anonymise metadata ({policy.label})") as op:
        written = apply_anonymise(plan, op=op)
    print(f"Anonymised {len(written)} file(s). The Editor's Undo takes it back, and keeps "
          "the originals under .bidsmgr/ until then: do not share that folder.")
    for path, why in plan.failed:
        print(f"  NOT anonymised, could not be rewritten: {path}: {why}")
    return 1 if plan.failed else 0


if __name__ == "__main__":
    sys.exit(main())
