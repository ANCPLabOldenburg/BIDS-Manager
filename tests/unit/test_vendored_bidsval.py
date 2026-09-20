"""bidsval is vendored, and BIDS Manager must reach only the vendored copy.

bidsval is BIDS Manager's own package and the single implementation of two
things the tool rests on: validation, and the interpretation of the BIDS
schema. Depending on it through PyPI meant a schema fix could not be tested
from here without publishing a release first.

Two properties matter. Nothing in ``bidsmgr`` may import the INSTALLED
package, or the two copies diverge silently on a machine that has both; and
the vendored copy has to be the whole thing, not the parts that happened to be
needed on the day it was copied.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "bidsmgr"
VENDORED = PACKAGE / "vendor" / "bidsval"


def _imports(path: Path) -> list[str]:
    """Every absolute module name imported by a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append(node.module or "")
    return out


def _reaches_installed(paths) -> list[str]:
    return [
        str(path)
        for path in paths
        for name in _imports(path)
        if name == "bidsval" or name.startswith("bidsval.")
    ]


def test_nothing_reaches_the_installed_package() -> None:
    """A machine with both copies must not quietly use the wrong one."""
    sources = [
        p for p in PACKAGE.rglob("*.py")
        if "vendor" not in p.relative_to(PACKAGE).parts
    ]
    assert not _reaches_installed(sources)


def test_the_vendored_copy_imports_itself_relatively() -> None:
    """Which is why it works at a new path unmodified. An absolute
    self-import would reach the INSTALLED package from inside the vendored
    one, mixing the two."""
    assert not _reaches_installed(VENDORED.rglob("*.py"))


def test_the_vendored_copy_is_importable_and_whole() -> None:
    from bidsmgr.vendor import bidsval

    assert bidsval.__version__
    for name in ("schema", "validate", "issues", "report", "rules", "context"):
        __import__(f"bidsmgr.vendor.bidsval.{name}")


def test_the_schema_namespace_is_there() -> None:
    """``bidsmgr/schema`` is a thin adapter over this. If the namespace went
    missing the adapter would silently fall back and BIDS facts would get
    derived twice, which is what guard 8 exists to prevent."""
    from bidsmgr.vendor.bidsval import schema as bv

    for name in (
        "sidecar_fields", "dataset_description_fields", "metadata_by_name",
        "short_to_long", "entity_pattern", "field_applies", "datatypes",
        "suffixes", "bids_version",
    ):
        assert hasattr(bv, name), name


def test_the_licence_travels_with_the_code() -> None:
    assert (VENDORED / "LICENSE").is_file()


def test_no_packaging_metadata_was_copied() -> None:
    """Source and licence only. ``__pycache__`` is not checked: Python makes
    it on import and it is gitignored."""
    junk = [
        str(p.relative_to(VENDORED)) for p in VENDORED.rglob("*")
        if p.name == "PKG-INFO" or p.name.endswith(".egg-info")
        or p.suffix in (".whl", ".tar.gz")
    ]
    assert not junk, junk


def test_the_adapter_still_answers_schema_questions() -> None:
    """The end the vendoring is for: BIDS facts still arrive, from one place."""
    from bidsmgr import schema

    assert schema.bids_version()
    assert len(schema.sidecar_fields("anat", "T1w")) > 50
    assert "sub" in schema.entity_keys()


def test_validation_still_runs_through_the_vendored_engine(
    tmp_path: Path,
) -> None:
    import json

    from bidsmgr.editor.validator import validate

    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"")
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    report = validate(root)
    assert report.files
    assert set(report.counts) >= {"ok", "warn", "err"}


@pytest.mark.parametrize("module", ["mne", "nibabel", "pandas", "yaml"])
def test_it_needs_no_dependency_we_did_not_already_ship(module: str) -> None:
    __import__(module)


# ---------------------------------------------------------------------------
# Proving it BEHAVIOURALLY, not just by reading the import statements
# ---------------------------------------------------------------------------
#
# The tests above parse the AST and look at import nodes. That is necessary
# and it is not sufficient, and the gap had teeth: the vendored schema
# resolver called
#
#     resources.files("bidsval.schema")
#
# which names the module in a STRING, so no import node existed for the AST
# walk to find. On a developer machine with bidsval pip-installed it resolved
# to the installed package and everything looked right, including these tests.
# On a clean Linux or Windows machine it raised ModuleNotFoundError before the
# GUI could start.
#
# The only test that could have caught it is one that makes the installed
# package genuinely unreachable and then does real work. That catches string
# imports, __import__, importlib, entry points and anything else, because it
# tests the outcome rather than the spelling.


class _Blocker:
    """Make top-level ``bidsval`` unimportable, as it is on a clean machine."""

    def find_spec(self, name, path=None, target=None):
        if name == "bidsval" or name.startswith("bidsval."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


@pytest.fixture
def without_installed_bidsval():
    """Run the body on a machine that does not have bidsval.

    ``sys.modules`` is purged as well as the meta path, because a module
    already imported is answered from the cache and never reaches a finder.
    The schema caches are cleared for the same reason: a resolve that was
    already done would not exercise the lookup under test.
    """
    import importlib
    import sys

    from bidsmgr.vendor.bidsval.schema import invalidate_dataset_cache

    # The MODULE, not the ``resolve`` function the package re-exports under
    # the same name.
    resolve_mod = importlib.import_module(
        "bidsmgr.vendor.bidsval.schema.resolve"
    )

    def drop_caches() -> None:
        resolve_mod._load.cache_clear()
        invalidate_dataset_cache()

    saved = {
        name: module for name, module in sys.modules.items()
        if name == "bidsval" or name.startswith("bidsval.")
    }
    for name in saved:
        del sys.modules[name]
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    drop_caches()
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        drop_caches()


def test_the_installed_package_really_is_unreachable(
    without_installed_bidsval,
) -> None:
    """Guard on the fixture itself. Without this, every test below could be
    passing because the block silently did nothing."""
    with pytest.raises(ModuleNotFoundError):
        __import__("bidsval")


def test_the_schema_resolves_with_no_bidsval_installed(
    without_installed_bidsval,
) -> None:
    """This is the exact call that crashed on Linux."""
    from bidsmgr.vendor.bidsval import schema as bv

    assert bv.resolve() is not None


def test_every_bundled_version_is_reachable(
    without_installed_bidsval,
) -> None:
    """Vendoring the code without the schema files ships a validator with
    nothing to validate against."""
    from bidsmgr.vendor.bidsval import schema as bv

    versions = bv.available_versions()
    assert len(versions) >= 5, versions
    for version in versions:
        assert bv.resolve(version) is not None, version


def test_the_adapter_answers_with_no_bidsval_installed(
    without_installed_bidsval,
) -> None:
    from bidsmgr.schema import sidecar_fields

    assert len(sidecar_fields("anat", "T1w")) > 50


def test_validation_runs_with_no_bidsval_installed(
    without_installed_bidsval, tmp_path: Path,
) -> None:
    import json

    from bidsmgr.editor.validator import validate

    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 64)
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0", "DatasetType": "raw"})
    )
    report = validate(root)
    assert report.files


# ---------------------------------------------------------------------------
# The data, and that it will actually be shipped
# ---------------------------------------------------------------------------


def test_the_bundled_schemas_were_copied_too() -> None:
    bundled = VENDORED / "schema" / "bundled"
    files = sorted(p.name for p in bundled.glob("*.json"))
    assert len(files) >= 5, files
    assert all((bundled / name).stat().st_size > 100_000 for name in files)


def test_the_wheel_will_carry_them() -> None:
    """Package data is opt-in. Source files are found automatically and JSON
    is not, so a correct tree still ships a broken wheel without this line."""
    # Read as text rather than parsed: ``tomllib`` is 3.11+, and the floor
    # here is 3.10.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert '"bidsmgr.vendor.bidsval.schema"' in text
    entry = text.split('"bidsmgr.vendor.bidsval.schema"', 1)[1].split("\n", 1)[0]
    assert "bundled/*.json" in entry, entry


def test_the_version_is_stated_not_looked_up() -> None:
    """``version("bidsval")`` reported whatever was INSTALLED, so the number
    described a different copy of the code than the one running."""
    from bidsmgr.vendor import bidsval

    assert bidsval.__version__ != "0.0.0"
    # And it records that this copy is AHEAD of the release it is named
    # after. Plain "0.1.1" would name a PyPI artefact that does not contain
    # the anyOf field typing this copy carries.
    base, _, local = bidsval.__version__.partition("+")
    assert base == "0.1.1", base
    assert local.startswith("bidsmgr."), (
        "the local delta in schema/fields.py has to show in the version"
    )
    # Code only: the comment above the fix quotes the old call by name.
    code = [
        line for line in (VENDORED / "__init__.py").read_text(encoding="utf-8")
        .splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if 'version("bidsval")' in line]
