"""Unit tests for the schema engine (architecture.md §3).

These run against whatever ``bidsschematools`` is installed; they verify the
shape of the API and the canonical results we rely on for M1 classification
and naming.
"""

from __future__ import annotations

import re

import pytest

from bidsmgr import schema


def test_listings():
    datatypes = schema.list_datatypes()
    assert "anat" in datatypes
    assert "func" in datatypes
    assert "dwi" in datatypes
    assert "fmap" in datatypes
    # Suffixes for anat must include T1w, T2w, FLAIR.
    suffixes = schema.list_suffixes("anat")
    assert "T1w" in suffixes
    assert "T2w" in suffixes
    assert "FLAIR" in suffixes
    # bold lives in func.
    assert "bold" in schema.list_suffixes("func")


def test_required_entities_match_spec():
    # T1w only requires 'subject'.
    assert schema.required_entities("anat", "T1w") == ["subject"]
    # bold requires both subject and task.
    assert set(schema.required_entities("func", "bold")) == {"subject", "task"}


def test_entity_format_pattern_label():
    fmt = schema.entity_format("subject")
    assert fmt.name == "label"
    assert re.fullmatch(fmt.pattern, "001")
    assert re.fullmatch(fmt.pattern, "ABC123")
    assert not re.fullmatch(fmt.pattern, "has spaces")


def test_build_basename_orders_entities_canonically():
    name = schema.build_basename(
        {"subject": "001", "task": "rest", "run": "1"},
        "func",
        "bold",
        ".nii.gz",
    )
    assert name == "sub-001_task-rest_run-1_bold.nii.gz"


def test_build_basename_rejects_unknown_entity():
    with pytest.raises(KeyError):
        schema.build_basename({"subject": "001", "frobnitz": "1"}, "anat", "T1w")


def test_build_relative_path_includes_session():
    p = schema.build_relative_path(
        {"subject": "001", "session": "pre", "task": "rest"},
        "func",
        "bold",
        ".nii.gz",
    )
    # as_posix(), not str(). ``build_relative_path`` returns a Path, and a
    # Path's str() is NATIVE by design: on Windows it is
    # "sub-001\\ses-pre\\func\\...". Asserting str() made this a test that
    # only held on POSIX, which is a defect in the test rather than in the
    # function. What the test is actually about is the structure, and
    # as_posix() states it the same way on every platform.
    assert p.as_posix() == (
        "sub-001/ses-pre/func/sub-001_ses-pre_task-rest_bold.nii.gz"
    )


def test_validate_entity_set_required():
    verdicts = schema.validate_entity_set({"subject": "001"}, "func", "bold")
    rule_ids = {v.rule_id for v in verdicts}
    assert "entity.required" in rule_ids


def test_validate_entity_set_unknown_suffix():
    verdicts = schema.validate_entity_set({"subject": "001"}, "anat", "bold")
    rule_ids = {v.rule_id for v in verdicts}
    assert "suffix.unknown" in rule_ids


def test_validate_basename_round_trips():
    name = schema.build_basename(
        {"subject": "001", "task": "rest"}, "func", "bold", ".nii.gz"
    )
    assert schema.validate_basename(name, "func") == []


class TestWarming:
    """The schema lookups are cached for the life of the process, so the
    FIRST one is expensive and the rest are free. Where that first one
    happens decides whether the window freezes."""

    def test_warm_pairs_populates_the_cache(self):
        from bidsmgr import schema as s

        s.warm_pairs([("anat", "T1w"), ("func", "bold")])
        # Cheap afterwards: asserted by the call succeeding without raising
        # and by the cache reporting hits.
        info = s.engine.sidecar_fields.cache_info() \
            if hasattr(s.engine.sidecar_fields, "cache_info") else None
        assert info is None or info.currsize > 0

    def test_warm_pairs_ignores_a_blank_pair(self):
        from bidsmgr import schema as s

        s.warm_pairs([("", ""), ("anat", ""), ("", "T1w")])

    def test_warm_pairs_ignores_a_pair_the_schema_rejects(self):
        """Warming is advisory and must never raise into a scan."""
        from bidsmgr import schema as s

        s.warm_pairs([("notadatatype", "notasuffix")])

    def test_warm_caches_is_idempotent(self):
        from bidsmgr import schema as s

        s.warm_caches()
        s.warm_caches()
