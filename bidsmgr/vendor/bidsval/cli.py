"""Command-line entry point for bidsval.

* ``bidsval validate PATH`` - validate a dataset and report errors and warnings as
  text, JSON, SARIF, or HTML.
* ``bidsval schema`` - show the schema version a selector resolves to and the bundled versions.
* ``bidsval eval EXPR`` - evaluate one BIDS schema expression against a context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .expr import EvaluationError, evaluate_string
from .issues import Severity
from .render import EXTENSIONS, RENDERERS
from .schema import SchemaNotAvailable, available_versions, bids_version, resolve, schema_version
from .validate import validate as run_validate

_DESCRIPTION = """\
A schema-driven, pydantic-typed, in-process BIDS validator written in pure Python.

It reads the official BIDS schema and checks a dataset against the rules in it: file
names and locations, sidecar metadata (presence and value type), associated files, and
tabular columns. It runs with no external runtime and reports findings as text, JSON,
SARIF, or HTML.
"""

_EPILOG = """\
Examples:
  bidsval validate /path/to/dataset
  bidsval validate /path/to/dataset --show all
  bidsval validate /path/to/dataset --schema 1.10.0
  bidsval validate /path/to/dataset --output-type json,html --out-dir ./reports
  bidsval schema
  bidsval eval "suffix == 'T1w'" --context '{"suffix": "T1w"}'

Run 'bidsval <command> -h' for command-specific help and more examples.

Exit codes:
  0   the dataset is valid (no errors); warnings may still be present
  1   validation found errors, or an expression failed to evaluate
  2   usage error, file/IO error, schema not available, or invalid JSON input
"""

_VALIDATE_DESCRIPTION = """\
Validate a BIDS dataset against the schema and report errors and warnings.

A dataset is valid when it has no errors. Errors are rule violations (a misplaced or
misnamed file, a required field that is missing, a value of the wrong type). Warnings
flag recommended-but-missing metadata and do not affect validity. Use --subject to
check a single participant.
"""

_VALIDATE_EPILOG = """\
Examples:
  # quick check (text summary; exits non-zero on errors, so it fits CI)
  bidsval validate /data/my_study

  # show everything, including warnings and suppressed notes
  bidsval validate /data/my_study --show all

  # check a single subject (the sub- prefix is optional)
  bidsval validate /data/my_study --subject 01

  # pin a schema version for reproducible results
  bidsval validate /data/my_study --schema 1.10.0

  # skip NIfTI header reading (faster)
  bidsval validate /data/my_study --no-headers

  # write every format into a directory
  bidsval validate /data/my_study --out-type all --out-dir ./reports

Output:
  One format prints to stdout (text by default). 'all' (or any comma-separated set)
  writes report.<ext> per format (report.txt, report.json, report.sarif, report.html)
  into --out-dir, or the current directory if --out-dir is omitted. In the text report
  each issue is one line: SEVERITY CODE [field] file - message.
"""

_SCHEMA_DESCRIPTION = """\
Show the schema bidsval would use and the versions bundled with this install.

Prints the BIDS version and the schema version a selector resolves to, plus the list of
bundled versions you can pass to --schema, here or on 'validate'.
"""

_SCHEMA_EPILOG = """\
Examples:
  # the default (bundled latest) schema
  bidsval schema

  # a specific bundled version
  bidsval schema --schema 1.10.0

  # the development tip, fetched from the spec and cached
  bidsval schema --schema latest
"""

_EVAL_DESCRIPTION = """\
Evaluate one BIDS schema expression against a context and print the result as JSON.

This exposes the same expression engine the validator uses, which is handy for
understanding a rule or testing a single condition. Undefined names evaluate to null.
"""

_EVAL_EPILOG = """\
Examples:
  # a simple comparison
  bidsval eval "suffix == 'T1w'" --context '{"suffix": "T1w"}'

  # combine conditions with && and ||
  bidsval eval "x > 0 && x < 10" --context '{"x": 5}'

  # arithmetic and the modulo operator
  bidsval eval "n % 2 == 0" --context '{"n": 4}'

Operators:
  comparison  ==  !=  <  <=  >  >=
  logical     &&  ||  !
  arithmetic  +  -  *  /  %

  --context must be a JSON object (not an array or a scalar). An undefined name is null.
"""


# Order subcommands are shown in the full (`bidsval --help`) dump: the primary
# command first.
_HELP_ORDER = ["validate", "schema", "eval"]


def _print_full_help(parser: argparse.ArgumentParser) -> None:
    """Print the overview help, then every subcommand's full help.

    Argparse's default top-level help lists only the subcommand names. This prints
    that overview and then each subcommand's complete help (its arguments, their
    explanations, and its examples), so ``bidsval --help`` shows everything in one
    place without drilling into each subcommand.
    """
    parser.print_help()
    sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    for action in sub_actions:
        names = sorted(
            action.choices,
            key=lambda n: _HELP_ORDER.index(n) if n in _HELP_ORDER else len(_HELP_ORDER),
        )
        for name in names:
            print("\n" + "=" * 78)
            print(f"  bidsval {name}")
            print("=" * 78)
            action.choices[name].print_help()


class _FullHelpAction(argparse.Action):
    """A ``-h/--help`` that prints the overview plus every subcommand's full help."""

    def __init__(
        self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None
    ):
        super().__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        _print_full_help(parser)
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bidsval",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action=_FullHelpAction,
        help="show this help, including every command's options and examples, and exit",
    )
    parser.add_argument("--version", action="version", version=f"bidsval {__version__}")
    subcommands = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    schema_cmd = subcommands.add_parser(
        "schema",
        help="show the resolved schema version and bundled versions",
        description=_SCHEMA_DESCRIPTION,
        epilog=_SCHEMA_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    schema_cmd.add_argument(
        "--schema",
        default=None,
        metavar="SELECTOR",
        help="schema to resolve: a BIDS version (e.g. 1.11.1), 'latest', a URL, a local "
        "schema.json, or a YAML schema source directory (default: the bundled latest)",
    )
    schema_cmd.set_defaults(func=_run_schema)

    eval_cmd = subcommands.add_parser(
        "eval",
        help="evaluate a BIDS schema expression",
        description=_EVAL_DESCRIPTION,
        epilog=_EVAL_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_cmd.add_argument(
        "expression", metavar="EXPR", help="the expression to evaluate, e.g. \"suffix == 'T1w'\""
    )
    eval_cmd.add_argument(
        "--context",
        default="{}",
        metavar="JSON",
        help="JSON object of variables the expression can reference (default: {}). "
        "Must be a JSON object, not an array or a scalar.",
    )
    eval_cmd.set_defaults(func=_run_eval)

    validate_cmd = subcommands.add_parser(
        "validate",
        help="validate a BIDS dataset",
        description=_VALIDATE_DESCRIPTION,
        epilog=_VALIDATE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_cmd.add_argument(
        "dataset",
        metavar="PATH",
        help="path to the dataset root (the folder that holds dataset_description.json)",
    )
    validate_cmd.add_argument(
        "--schema",
        default=None,
        metavar="SELECTOR",
        help="schema to validate against: a BIDS version (e.g. 1.11.1), 'latest', a URL, a "
        "local schema.json, or a YAML schema source directory (default: the bundled latest). "
        "Run 'bidsval schema' to list bundled versions.",
    )
    validate_cmd.add_argument(
        "--subject",
        default=None,
        metavar="SUB",
        help="validate only this subject. Accepts sub-01 or just 01 (the sub- prefix is "
        "added if missing).",
    )
    validate_cmd.add_argument(
        "--no-headers",
        action="store_true",
        help="skip NIfTI header checks (faster). Headers are read by default (needs nibabel); "
        "if nibabel is not installed these checks are skipped anyway.",
    )
    validate_cmd.add_argument(
        "--recursive",
        action="store_true",
        help="also validate BIDS datasets under derivatives/ (each on its own).",
    )
    validate_cmd.add_argument(
        "--out-type",
        "--output-type",
        dest="output_type",
        default="text",
        metavar="TYPES",
        help="output format: text, json, sarif, html, or 'all' (default: text). A single "
        "format prints to stdout unless --out-dir is given; 'all' (and any comma-separated "
        "set) writes report.<ext> files into --out-dir, or the current directory if --out-dir "
        "is omitted.",
    )
    validate_cmd.add_argument(
        "--out-dir",
        metavar="DIR",
        help="directory to write the report file(s) into, one report.<ext> per format "
        "(created if needed). Without it, a single format prints to stdout and 'all' writes "
        "into the current directory.",
    )
    validate_cmd.add_argument(
        "--show",
        default="all",
        metavar="LEVELS",
        help="severities to display: any of error, warning, ignore, or 'all' (default: all). "
        "Filters the output only; it does not change validity or the exit code.",
    )
    validate_cmd.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        metavar="N",
        help="maximum number of TSV rows scanned per table for value checks (default: 1000).",
    )
    validate_cmd.add_argument(
        "--filenames-only",
        action="store_true",
        help="report only file name and path findings (skip content findings in the output).",
    )
    validate_cmd.add_argument(
        "--list-schemas",
        action="store_true",
        help="list the default and bundled schema versions, then exit.",
    )
    validate_cmd.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the bidsval version before validating.",
    )
    validate_cmd.set_defaults(func=_run_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        _print_full_help(parser)
        return 0
    return args.func(args)


def _run_schema(args: argparse.Namespace) -> int:
    try:
        schema = resolve(args.schema)
    except SchemaNotAvailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"BIDS version  : {bids_version(schema)}")
    print(f"schema version: {schema_version(schema)}")
    print(f"bundled       : {', '.join(available_versions())}")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    try:
        context = json.loads(args.context)
    except json.JSONDecodeError as error:
        print(f"error: --context is not valid JSON: {error}", file=sys.stderr)
        return 2
    if not isinstance(context, dict):
        print("error: --context must be a JSON object", file=sys.stderr)
        return 2
    try:
        result = evaluate_string(args.expression, context)
    except EvaluationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


# Codes produced by the file name / path rules, for ``--filenames-only``.
_FILENAME_CODES = frozenset(
    {
        "NOT_INCLUDED",
        "FILENAME_MISMATCH",
        "EXTENSION_MISMATCH",
        "DATATYPE_MISMATCH",
        "ENTITY_NOT_IN_RULE",
        "MISSING_REQUIRED_ENTITY",
        "INVALID_ENTITY_LABEL",
        "ENTITY_WITH_NO_LABEL",
        "ALL_FILENAME_RULES_HAVE_ISSUES",
        "INVALID_LOCATION",
    }
)


def _list_schemas() -> int:
    """Print the default and bundled schema versions (the ``--list-schemas`` flag)."""
    default = resolve(None)
    print(f"default: BIDS {bids_version(default)} (schema {schema_version(default)})")
    print(f"bundled: {', '.join(available_versions())}")
    return 0


def _keep_filename_findings(report: object) -> None:
    """Drop every non-filename finding from ``report`` in place (``--filenames-only``)."""
    report.dataset_issues.issues = [
        i for i in report.dataset_issues.issues if i.code in _FILENAME_CODES
    ]
    for verdict in report.files:
        verdict.issues = [i for i in verdict.issues if i.code in _FILENAME_CODES]
        verdict.recompute_severity()
    report.files = [v for v in report.files if v.issues]


def _run_validate(args: argparse.Namespace) -> int:
    if args.list_schemas:
        return _list_schemas()
    if args.verbose:
        print(f"bidsval {__version__}", file=sys.stderr)
    subjects = None
    if args.subject:
        sub = args.subject if args.subject.startswith("sub-") else f"sub-{args.subject}"
        subjects = [sub]
    try:
        report = run_validate(
            args.dataset,
            schema=args.schema,
            read_headers=not args.no_headers,
            max_rows=args.max_rows,
            subjects=subjects,
            recursive=args.recursive,
        )
    except SchemaNotAvailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        types = _parse_output_types(args.output_type)
        severities = _parse_severities(args.show)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # Findings are filtered for display only; validity always depends on errors.
    display = report.filtered(severities)
    if args.filenames_only:
        _keep_filename_findings(display)

    # A single format prints to stdout (composable). 'all' and any multi-format set
    # write report.<ext> files: into --out-dir if given, otherwise the current
    # directory, since several documents cannot stream to stdout.
    if args.out_dir or len(types) > 1:
        out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        for output_type in sorted(types):
            destination = out_dir / f"report.{EXTENSIONS[output_type]}"
            destination.write_text(RENDERERS[output_type](display), encoding="utf-8")
            print(f"wrote {destination}", file=sys.stderr)
    else:
        print(RENDERERS[next(iter(types))](display))

    return 0 if report.is_valid else 1


def _parse_output_types(value: str) -> set[str]:
    requested = [t.strip().lower() for t in value.split(",") if t.strip()]
    if "all" in requested:
        return set(RENDERERS)
    unknown = [t for t in requested if t not in RENDERERS]
    if unknown:
        raise ValueError(
            f"unknown --output-type {unknown}; choose from {sorted(RENDERERS)} or 'all'"
        )
    return set(requested) or {"text"}


def _parse_severities(value: str) -> set[Severity]:
    requested = [s.strip().lower() for s in value.split(",") if s.strip()]
    if "all" in requested:
        return set(Severity)
    out: set[Severity] = set()
    for name in requested:
        try:
            out.add(Severity(name))
        except ValueError as error:
            raise ValueError(
                f"unknown --show level {name!r}; choose from error, warning, ignore, or 'all'"
            ) from error
    return out or {Severity.ERROR, Severity.WARNING}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
