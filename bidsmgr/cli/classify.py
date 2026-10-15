"""``bidsmgr-classify``: name an inventory's MRI series from their
acquisition parameters again, and say why.

Reads the features the scan cached beside the inventory
(``<inventory>.tsv.series_features.json.gz``), so no DICOM is read: a
changed setting is tried in a moment. Prints, per series, the name it has,
the name the parameters give, how sure they are, who decided, and the
strongest evidence. ``--apply`` writes the new names (and the
classification columns) into the inventory.

    bidsmgr-classify inventory.tsv
    bidsmgr-classify inventory.tsv --config my_classifier.json --apply
    bidsmgr-classify --print-config > my_classifier.json
    bidsmgr-classify --print-rulebook > rules.json

Every number the classifier tests is in the settings; the evidence rules
(which feature counts for which name, and how much) are the rulebook. A
settings file's ``"rulebook"`` holds changes to it:
``{"candidates": {"anat/T2w": {...}}}`` replaces one name's rules, ``null``
removes a name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from ..classifier.acquisition import cache
from ..classifier.acquisition import config as cfgmod
from ..classifier.acquisition import engine
from ..classifier.acquisition import scan as by_parameters


def load_config(path: Optional[Path]) -> cfgmod.ClassifierConfig:
    """The settings in ``path`` (the defaults for ``None``). Raises
    ``ValueError`` saying what is wrong when its rule changes cannot be
    used."""
    if path is None:
        return cfgmod.ClassifierConfig()
    config = cfgmod.load_json(Path(path).read_text(encoding="utf-8"))
    if config.rulebook:
        problems = engine.rulebook_problems(engine.rulebook(config.rulebook))
        if problems:
            raise ValueError("its rulebook changes cannot be used: " + "; ".join(problems[:5]))
    return config


def run_classify(tsv: Path, *, config: Optional[cfgmod.ClassifierConfig] = None,
                 apply: bool = False, out=sys.stdout, as_json: bool = False) -> int:
    """Score ``tsv`` again; print the report; write it with ``apply``.
    Returns the number of series whose name changes."""
    config = config or cfgmod.ClassifierConfig()
    tsv = Path(tsv)
    features = cache.read(tsv)
    if not features:
        print(f"{tsv.name} has no acquisition features beside it "
              f"({cache.features_path(tsv).name}): scan the raw data again.", file=out)
        return 0
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    changes = by_parameters.rescore(df, features, config)
    renamed = [c for c in changes if c.renamed]
    if as_json:
        print(json.dumps([{
            "row": c.index, "description": str(df.at[c.index, "sequence"]),
            "now": c.old, "parameters": c.new, "check": c.check,
            "confidence": float(c.columns["classification_confidence"]),
            "evidence": json.loads(c.columns["classification_evidence"]),
        } for c in changes], indent=2), file=out)
    else:
        for c in changes:
            ev = json.loads(c.columns["classification_evidence"])
            mark = "->" if c.renamed else "  "
            flag = " (check)" if c.check else ""
            print(f"{mark} {str(df.at[c.index, 'sequence'])[:44]:44s} {c.old or '-':22s} "
                  f"{(c.new if c.new != c.old else 'entities') if c.renamed else '':22s} "
                  f"{c.columns['classification_confidence']} {ev.get('decided_by') or '-'}{flag}",
                  file=out)
            for line in ev.get("for", [])[:3]:
                print(f"      {line['points']:+.1f}  {line['text']}", file=out)
        print(f"{len(changes)} series scored, {len(renamed)} named differently "
              f"(mode {config.general.mode}).", file=out)
    if apply and config.general.mode != "off":
        if config.general.mode == "shadow":
            renamed_frame, _n = by_parameters.apply_to_frame(
                df, [by_parameters.Change(c.index, c.old, c.old, {}, False, c.columns, c.check)
                     for c in changes])
        else:
            renamed_frame, _n = by_parameters.apply_to_frame(df, changes)
        renamed_frame.to_csv(tsv, sep="\t", index=False)
        print(f"Written: {tsv}", file=out)
    return len(renamed)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-classify",
        description="Name an inventory's MRI series from their acquisition parameters "
                    "again (no DICOM is read) and say why.")
    parser.add_argument("inventory", nargs="?", type=Path, help="The inventory TSV")
    parser.add_argument("--config", type=Path, default=None,
                        help="The classifier settings (JSON; --print-config writes one)")
    parser.add_argument("--mode", choices=("applied", "shadow", "off"), default=None,
                        help="Override the settings' mode: name the series (applied) or "
                             "only report (shadow)")
    parser.add_argument("--apply", action="store_true",
                        help="Write the names and the classification columns into the "
                             "inventory")
    parser.add_argument("--json", action="store_true", help="The report as JSON")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the default settings as JSON and stop")
    parser.add_argument("--print-rulebook", action="store_true",
                        help="Print the evidence rules in use (with --config: with its "
                             "changes) as JSON and stop")
    args = parser.parse_args(argv)
    if args.print_config:
        print(cfgmod.ClassifierConfig().model_dump_json(indent=2))
        return 0
    if args.print_rulebook:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            parser.error(f"--config could not be read: {exc}")
        print(json.dumps(engine.rulebook(config.rulebook), indent=2, ensure_ascii=False))
        return 0
    if args.inventory is None:
        parser.error("give the inventory TSV (or --print-config)")
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(f"--config could not be read: {exc}")
    if args.mode:
        config.general.mode = args.mode
    run_classify(args.inventory, config=config, apply=args.apply, as_json=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
