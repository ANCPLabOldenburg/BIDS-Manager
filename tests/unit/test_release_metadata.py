"""The release metadata at the repository's root agrees with the package.

``CITATION.cff`` is what GitHub's "Cite this repository" and Zenodo read, and
``codemeta.json`` is what the EBRAINS Knowledge Graph reads: it checks the file
regularly and registers a new version of BIDS Manager whenever its ``version``
changes. Both state the version, the authors and their ORCIDs by hand, so a
release that bumps ``pyproject.toml`` alone would leave them describing the
previous one. These tests fail until all of them say the same thing.
"""

from __future__ import annotations

import json
import re
from importlib.metadata import requires
from pathlib import Path

import pytest
import yaml

import bidsmgr

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/ANCPLabOldenburg/BIDS-Manager"
ORCIDS = {"López Vilaret": "https://orcid.org/0009-0007-0504-5258",
          "Rieger": "https://orcid.org/0000-0003-0955-2306"}


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def codemeta() -> dict:
    return json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))


def test_every_file_states_the_package_version(citation, codemeta) -> None:
    version = _pyproject_version()
    assert str(citation["version"]) == version
    assert codemeta["version"] == version
    # The download is that version's source archive on PyPI, and the release
    # notes are that version's card on the documentation's Updates page.
    assert codemeta["downloadUrl"].endswith(f"bids_manager-{version}.tar.gz")
    assert codemeta["releaseNotes"].endswith("#v" + version.replace(".", ""))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"BIDS Manager (version {version})" in readme


def test_the_installed_package_is_the_one_described() -> None:
    assert bidsmgr.__version__ == _pyproject_version()


def test_the_authors_and_their_orcids_agree(citation, codemeta) -> None:
    cited = {a["family-names"]: a["orcid"] for a in citation["authors"]}
    meta = {a["familyName"]: a["@id"] for a in codemeta["author"]}
    assert cited == ORCIDS
    assert meta == ORCIDS


def test_both_point_at_the_repository_and_the_licence(citation, codemeta) -> None:
    assert citation["repository-code"] == REPO
    assert codemeta["codeRepository"] == REPO
    assert citation["license"] == "MIT"
    assert codemeta["license"] == "https://spdx.org/licenses/MIT"
    assert (ROOT / "LICENSE").is_file()


def test_codemeta_lists_every_runtime_requirement(codemeta) -> None:
    """One entry per dependency of the package, with its version range."""
    declared = [r for r in (requires("bids-manager") or []) if "extra ==" not in r]
    names = {re.split(r"[<>=!~;\s]", r, maxsplit=1)[0].lower() for r in declared}
    listed = {re.split(r"[<>=!~;\s(]", r, maxsplit=1)[0].lower()
              for r in codemeta["softwareRequirements"]}
    assert listed == names
