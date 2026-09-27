"""``recording-<label>`` has to be a label the standard accepts.

The signal name read out of a recording went straight into the filename, and
a source calls a channel whatever it likes. A Siemens PMU dump yields
``external_trigger``, which produced::

    sub-001_task-x_recording-external_trigger_physio.tsv.gz

That is not a BIDS name, and it does not merely fail validation: the
underscore is the entity SEPARATOR, so every tool that parses BIDS names
reads an entity ``recording-external`` followed by a stray token, and sees a
different file from the one that was written.

Fixed in the vendored ``bidsphysio`` rather than in a fixup afterwards,
because the writer is the thing choosing the name, and because upstream is
no longer maintained.

**Which characters, per BIDS version.** Checked here against every schema
the installed ``bidsval`` ships rather than against a constant, because the
answer has changed once already:

    1.8.0, 1.9.0, 1.10.0      [0-9a-zA-Z]+
    1.10.1, 1.11.0, 1.11.1    [0-9a-zA-Z+]+

The plus sign arrived in 1.10.1 and means something specific there
(concatenating several applicable labels), so it is not a separator to reach
for. Alphanumeric-only is the one spelling valid under every version, which
is what the sanitiser produces, and these tests hold it to that against all
of them.
"""

from __future__ import annotations

import re

import pytest

from bidsmgr.vendor.bidsphysio.base.bidsphysio import (
    bids_label,
    unique_bids_labels,
)


def _patterns() -> dict[str, str]:
    """``recording`` entity pattern per BIDS version, from bidsval."""
    import bidsval.schema as bs

    out: dict[str, str] = {}
    for version in sorted(bs.available_versions()):
        try:
            selector = bs.resolve(version)
            pattern = bs.entity_pattern(selector, "recording")
        except Exception:  # noqa: BLE001 - a version we cannot reach
            continue
        if pattern:
            out[version] = pattern
    return out


class TestWhatItProduces:
    @pytest.mark.parametrize("raw,expected", [
        # The reported one. The word boundary the underscore marked is kept
        # by capitalising what followed, so the label still reads.
        ("external_trigger", "externalTrigger"),
        ("resp-belt", "respBelt"),
        ("ECG II", "ECGII"),
        ("100 Hz", "100Hz"),
        ("x  y", "xY"),
        # Already valid: left exactly as it is.
        ("trigger", "trigger"),
        ("PULS", "PULS"),
        ("cardiac", "cardiac"),
    ])
    def test_it_cleans_a_name_without_losing_the_words(self, raw, expected):
        assert bids_label(raw) == expected

    @pytest.mark.parametrize("raw", ["---", "", "   ", "___", None])
    def test_a_name_with_nothing_usable_falls_back(self, raw):
        """``recording-`` with no value is a worse name than one that says
        little."""
        assert bids_label(raw) == "signal"

    def test_the_fallback_can_be_chosen(self):
        assert bids_label("!!!", fallback="channel") == "channel"


class TestEveryBidsVersionAccepts:
    def test_bidsval_offers_more_than_one_version_to_check(self):
        assert len(_patterns()) >= 2, (
            "cannot check across versions; the claim in the module "
            "docstring is unverified"
        )

    @pytest.mark.parametrize("raw", [
        "external_trigger", "resp-belt", "ECG II", "100 Hz", "trigger",
        "PULS", "---", "a/b\\c", "x.y", "über", "こんにちは", "a+b",
    ])
    def test_the_label_is_valid_under_all_of_them(self, raw):
        label = bids_label(raw)
        for version, pattern in _patterns().items():
            assert re.fullmatch(pattern, label), (
                f"{label!r} from {raw!r} is not a valid recording label "
                f"under BIDS {version} ({pattern})"
            )

    def test_the_plus_sign_is_not_used(self):
        """It is legal from 1.10.1 and means "several applicable labels",
        so reaching for it as a separator would be both wrong in meaning
        and invalid on three of the six versions."""
        assert "+" not in bids_label("a+b")


class TestCollisions:
    def test_two_names_cleaning_to_one_are_broken_apart(self):
        """Two recordings with one name means the second overwrites the
        first, which loses data silently."""
        assert unique_bids_labels(["ecg lead", "ecg_lead", "ecg-lead"]) == [
            "ecgLead", "ecgLead2", "ecgLead3",
        ]

    def test_distinct_names_are_left_alone(self):
        assert unique_bids_labels(["cardiac", "respiratory"]) == [
            "cardiac", "respiratory",
        ]

    def test_the_disambiguated_labels_are_still_valid(self):
        for label in unique_bids_labels(["a b", "a_b", "a-b"]):
            for pattern in _patterns().values():
                assert re.fullmatch(pattern, label)
