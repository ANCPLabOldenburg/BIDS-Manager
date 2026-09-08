"""Switching the BIDS version changes what BIDS Manager asks for.

The setting that picks a version existed and reached the validator alone. Every
form, table and fixup went on describing whatever version the install shipped,
so a dataset could be checked against one version while being filled in against
another, and nobody could tell from the screen.

These pin two versions that genuinely differ and assert the difference reaches
each surface. They are not testing bidsval's schemas: they are testing that BIDS
Manager asks its questions of the version in force.
"""

from __future__ import annotations

import pytest

from bidsmgr import schema as schema_mod
from bidsmgr.metadata.template_plan import build_template_tree, sidecar_section

# Far enough apart to differ, both bundled. 1.11.1 is the current default.
OLD = "1.8.0"
NEW = "1.11.1"


@pytest.fixture(autouse=True)
def _restore_default_version():
    """Never leak a pinned version into another test."""
    yield
    schema_mod.set_active_version(None)


def _at(version, call):
    schema_mod.set_active_version(version)
    return call()


def test_both_versions_are_available() -> None:
    available = schema_mod.available_versions()
    assert OLD in available and NEW in available


def test_the_version_in_force_is_the_one_reported() -> None:
    assert _at(OLD, schema_mod.bids_version) == OLD
    assert _at(NEW, schema_mod.bids_version) == NEW


def test_a_sidecar_declares_different_fields_per_version() -> None:
    old = {f.name for f in _at(OLD, lambda: schema_mod.sidecar_fields("anat", "T1w"))}
    new = {f.name for f in _at(NEW, lambda: schema_mod.sidecar_fields("anat", "T1w"))}
    assert old != new
    # Named, so a schema change that happens to keep the counts equal still
    # fails rather than passing by coincidence.
    assert "BodyPart" in new and "BodyPart" not in old
    assert "PhaseEncodingDirection" in old and "PhaseEncodingDirection" not in new


def test_field_applies_follows_the_version() -> None:
    assert _at(NEW, lambda: schema_mod.field_applies("BodyPart", "anat", "T1w"))
    assert not _at(OLD, lambda: schema_mod.field_applies("BodyPart", "anat", "T1w"))


def test_the_vocabulary_follows_the_version() -> None:
    """Not just sidecar fields: datatypes and entities come from the same schema."""
    old = set(_at(OLD, schema_mod.list_datatypes))
    new = set(_at(NEW, schema_mod.list_datatypes))
    assert new - old  # 1.11 added datatypes 1.8 did not have
    assert old <= new


def test_the_template_a_user_sees_follows_the_version() -> None:
    """The gate: the form itself, not only the layer beneath it."""
    def anat_fields(version):
        return {
            f.name
            for f in _at(version, lambda: sidecar_section("anat", "T1w")).fields
        }

    assert "BodyPart" not in anat_fields(OLD)
    # In 1.11 dcm2niix supplies BodyPart, so the template does not ask for it;
    # what matters is that the two versions disagree about the section at all.
    assert anat_fields(OLD) != anat_fields(NEW)


def test_the_whole_tree_follows_the_version() -> None:
    def names(version):
        tree = _at(version, lambda: build_template_tree([("anat", "T1w")]))
        return {
            f.name
            for node in tree
            for n in node.walk()
            if n.section
            for f in n.section.fields
        }

    assert names(OLD) != names(NEW)


def test_switching_back_restores_the_default() -> None:
    """Memoised answers are answers about one version, so they have to go."""
    before = {f.name for f in schema_mod.sidecar_fields("anat", "T1w")}
    schema_mod.set_active_version(OLD)
    assert {f.name for f in schema_mod.sidecar_fields("anat", "T1w")} != before
    schema_mod.set_active_version(None)
    assert {f.name for f in schema_mod.sidecar_fields("anat", "T1w")} == before
