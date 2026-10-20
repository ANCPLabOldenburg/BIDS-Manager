"""What anonymising a dataset changes: categories by purpose, each on or off
with its options, and two presets that set them.

Every choice follows the standard where it has one: dates are SHIFTED rather
than removed ("shifting is RECOMMENDED over removal"), into 1925 or earlier,
by the same number of days for every date of one participant, so intervals
survive; ages are capped at 89. Where the standard says nothing, the
Recommended preset keeps what analyses need (times of day, the site name)
and Strict keeps as little as still makes a valid dataset.

A category is one PURPOSE of a change (a person's name, a date, the site), so
a user decides "keep the dates, remove the names" rather than field by field;
every change a plan lists says which category it belongs to.

Pure data. ``bidsmgr.anonymise.plan`` reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Literal, Optional

OFF = "off"
RECOMMENDED = "recommended"
STRICT = "strict"
#: A policy that matches no preset.
CUSTOM = "custom"


@dataclass(frozen=True)
class Category:
    """One purpose of a change, as a user turns it on or off."""

    key: str
    title: str
    #: The heading it is listed under (:data:`PURPOSES`).
    purpose: str
    #: What it changes, in a sentence.
    summary: str
    #: ``(value, label, tooltip)`` of its options, the first the default.
    options: tuple[tuple[str, str, str], ...] = ()


#: The headings the categories are grouped under, in order.
PURPOSES: tuple[tuple[str, str], ...] = (
    ("who", "The participants"),
    ("when", "When it happened"),
    ("where", "The site and its equipment"),
    ("files", "Traces in files"),
)

CATEGORIES: tuple[Category, ...] = (
    Category(
        "identity", "Names and IDs", "who",
        "Names, IDs, birth dates and contact details, the scanner's copy of sex, age, "
        "weight and height, and the names of the people who ran the session: in "
        "sidecars, participants.tsv and recording headers."),
    Category(
        "ages", "Ages", "who",
        "Ages in participants.tsv: an exact age, or one over 89, can identify.",
        (("cap", "Over 89 as 89", "The standard's rule: every age over 89 written as 89."),
         ("bins", "Five-year bins",
          "Every age rounded down to a five-year bin (and capped at 89)."))),
    Category(
        "dates", "Dates and times", "when",
        "Every date of a participant moved back by one number of whole weeks, into "
        "1902-1925, so intervals and weekdays are kept: sidecars, scans and sessions "
        "tables, marker files, recording headers.",
        (("keep", "Keep times of day",
          "Times of day are kept: analyses of time of day still work."),
         ("midnight", "Times to midnight",
          "Every time of day set to midnight, and the scanner's own time fields "
          "removed."))),
    Category(
        "site", "The site", "where",
        "The institution, written as n/a (the standard recommends the fields, so they "
        "stay, saying nothing).",
        (("address", "Address only", "The institution's address; its name is kept."),
         ("all", "Name and address",
          "The institution's name, department and address."))),
    Category(
        "devices", "Devices", "where",
        "Device serial numbers and station names, replaced by stable pseudonyms "
        "(device-..., station-...), the same for the same machine everywhere."),
    Category(
        "uids", "DICOM UIDs", "where",
        "Study, series and instance UIDs, replaced by stable pseudonyms (2.25....): "
        "many systems build them from the scanner and the time."),
    Category(
        "traces", "Traces of the conversion", "files",
        "The time and original file name in every gzip header, and paths on the "
        "converting machine (OriginalFile)."),
    Category(
        "photos", "Photographs", "files",
        "EXIF and text metadata of JPEG and PNG pictures (dates, places, devices); the "
        "pixels are untouched."),
)

#: The categories by key.
CATEGORY: dict[str, Category] = {c.key: c for c in CATEGORIES}

#: What a change that only records the anonymisation belongs to
#: (``DeidentificationMethod``): not a category anyone turns off.
RECORD = "record"
RECORD_TITLE = "Record of what was done"


@dataclass(frozen=True)
class AnonymisePolicy:
    """One answer per category."""

    name: str = RECOMMENDED
    #: Names, IDs, birth dates, contact details, the people of the session.
    identity: bool = True
    #: Ages in participants.tsv: capped at 89, also binned, or kept.
    ages: Literal["cap", "bins", "keep"] = "cap"
    #: Shift every date of a participant by one number of whole weeks, into
    #: 1902-1925 ("shift"), or leave dates alone ("keep").
    dates: Literal["shift", "keep"] = "shift"
    #: The time of day: kept, or set to midnight (and the scanner's own time
    #: fields removed, the NIfTI header's "Time=" too).
    times: Literal["keep", "midnight"] = "keep"
    #: The institution: "address" its address only; "all" name, department
    #: and address; "keep" nothing.
    institution: Literal["address", "all", "keep"] = "address"
    #: Device serial numbers and station names: pseudonymised.
    devices: Literal["pseudonymise", "keep"] = "pseudonymise"
    #: DICOM UIDs: replaced by a pseudonym that is the same for the same UID
    #: everywhere (they embed the scanner and the time on many systems).
    uids: Literal["pseudonymise", "keep"] = "pseudonymise"
    #: The gzip header of every .nii.gz and .tsv.gz (its time stamp and the
    #: original file name), and paths on the converting machine.
    traces: bool = True
    #: Photographs: their EXIF and text metadata removed, the image untouched.
    photos: bool = True

    # -- by category ---------------------------------------------------------

    def enabled(self, key: str) -> bool:
        """Whether category ``key`` changes anything."""
        return {
            "identity": self.identity,
            "ages": self.ages != "keep",
            "dates": self.dates == "shift",
            "site": self.institution != "keep",
            "devices": self.devices == "pseudonymise",
            "uids": self.uids == "pseudonymise",
            "traces": self.traces,
            "photos": self.photos,
        }[key]

    def option(self, key: str) -> Optional[str]:
        """The option category ``key`` has chosen (None when it has none)."""
        return {"ages": self.ages if self.ages != "keep" else "cap",
                "dates": self.times,
                "site": self.institution if self.institution != "keep" else "address",
                }.get(key)

    def with_category(self, key: str, on: bool,
                      option: Optional[str] = None) -> "AnonymisePolicy":
        """This policy with category ``key`` on (with ``option``, else the one
        it had) or off; named after the preset it now matches, else custom."""
        if key not in CATEGORY:
            raise ValueError(f"unknown anonymisation category {key!r}; "
                             f"choose from {', '.join(CATEGORY)}")
        option = option or self.option(key)
        changes = {
            "identity": {"identity": on},
            "ages": {"ages": option if on else "keep"},
            "dates": {"dates": "shift" if on else "keep",
                      "times": option if on else "keep"},
            "site": {"institution": option if on else "keep"},
            "devices": {"devices": "pseudonymise" if on else "keep"},
            "uids": {"uids": "pseudonymise" if on else "keep"},
            "traces": {"traces": on},
            "photos": {"photos": on},
        }[key]
        return replace(self, **changes).named()

    def named(self) -> "AnonymisePolicy":
        """This policy named after the preset it matches, else custom."""
        for name, preset in PRESETS.items():
            if _same(self, preset):
                return replace(self, name=name)
        return replace(self, name=CUSTOM)

    def categories_on(self) -> list[str]:
        return [c.key for c in CATEGORIES if self.enabled(c.key)]

    @property
    def label(self) -> str:
        return PRESET_LABELS.get(self.name, "Custom")

    def describe(self) -> list[str]:
        """What this policy does, one line each, for a dialog or a log."""
        out = []
        if self.dates == "shift":
            out.append("Dates shifted into 1902-1925, by one number of whole weeks per "
                       "participant, so intervals and weekdays are kept")
        if self.times == "midnight":
            out.append("Times of day set to midnight; the scanner's own time fields "
                       "removed")
        if self.identity:
            out.append("Names, IDs, birth dates and the people who ran the session "
                       "removed, in sidecars, participants.tsv and recording headers")
        if self.uids == "pseudonymise":
            out.append("DICOM UIDs replaced by stable pseudonyms")
        if self.institution == "address":
            out.append("The institution's address written as n/a (its name kept)")
        elif self.institution == "all":
            out.append("The institution's name, department and address written as n/a")
        if self.devices == "pseudonymise":
            out.append("Device serial numbers and station names pseudonymised")
        if self.ages == "cap":
            out.append("Ages over 89 written as 89")
        elif self.ages == "bins":
            out.append("Ages rounded down to five-year bins, and capped at 89")
        if self.traces:
            out.append("The time stamp and original name in every gzip header, and "
                       "paths on the converting machine, removed")
        if self.photos:
            out.append("Photographs' EXIF and text metadata removed")
        return out


def _same(a: AnonymisePolicy, b: AnonymisePolicy) -> bool:
    return all(getattr(a, f.name) == getattr(b, f.name)
               for f in fields(AnonymisePolicy) if f.name != "name")


PRESETS: dict[str, AnonymisePolicy] = {
    RECOMMENDED: AnonymisePolicy(),
    STRICT: AnonymisePolicy(name=STRICT, times="midnight", institution="all", ages="bins"),
}

#: The choices a conversion's post step offers (custom is the Editor's).
PRESET_LABELS: dict[str, str] = {
    OFF: "Off",
    RECOMMENDED: "Recommended",
    STRICT: "Strict",
}


def policy_for(name: str, *, keep: tuple[str, ...] = ()) -> AnonymisePolicy | None:
    """The preset called ``name`` with the categories in ``keep`` turned off;
    ``None`` for off."""
    if name == OFF or not name:
        return None
    try:
        policy = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown anonymisation preset {name!r}; "
                         f"choose from {', '.join(PRESET_LABELS)}") from None
    for key in keep:
        policy = policy.with_category(key, False)
    return policy


__all__ = ["AnonymisePolicy", "CATEGORIES", "CATEGORY", "CUSTOM", "Category", "OFF",
           "PRESETS", "PRESET_LABELS", "PURPOSES", "RECOMMENDED", "RECORD", "RECORD_TITLE",
           "STRICT", "policy_for"]
