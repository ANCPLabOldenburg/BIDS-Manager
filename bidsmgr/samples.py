"""The sample datasets BIDS Manager publishes: ONE list.

Home's sample menu and the test suite's download fixture
(``tests/fixtures/sample_data.py``) both read it. They used to keep a list
each, and when every share on the cloud was reissued in October 2026 only
Home's was updated: the tests' links answered 404, and the tests that need
real data skipped everywhere a copy was not cached, as if offline. The
documentation's Sample data page (``site.yml`` there) holds the same links
and is checked against this list by hand when a dataset changes.

An installed BIDS Manager cannot be corrected, so a share must KEEP its link:
a new version of a dataset replaces the zip INSIDE its share, and ``version``
here is bumped so cached copies are fetched again. When a version gets a share
of its own instead, the old share must stay up: the multimodal sample v3 has
its own share (2026-10-26) and installed copies still download v2 from theirs.

Qt-free, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The documentation page that says what each dataset demonstrates.
SAMPLE_DATA_PAGE = (
    "https://ancplaboldenburg.github.io/bids_manager_documentation/start/sample-data/"
)


@dataclass(frozen=True)
class Sample:
    """One published dataset."""

    key: str            # how code refers to it ("multimodal", "pet", ...)
    label: str          # Home's menu entry
    token: str          # the cloud.uol.de share
    megabytes: int      # the zip, measured, so a large download is a visible one
    version: str        # the zip inside the share; bump when it is replaced

    @property
    def url(self) -> str:
        return f"https://cloud.uol.de/s/{self.token}/download"


#: In Home's menu order. Sizes measured 2026-10-26.
SAMPLES: tuple[Sample, ...] = (
    Sample("mri", "MRI walkthrough dataset", "qpE8KeBpqgTn2LB", 347, "1"),
    Sample("mri_advanced", "Advanced MRI (Siemens) dataset", "KdFzsDtLA9ijNAb", 4694, "1"),
    Sample("pet", "PET, DICOM and ECAT with blood", "Sfmdo6m2GCrKyfi", 10, "1"),
    Sample("eeg", "EEG motor-imagery dataset", "8YcmPwFJ5qzfEAq", 37, "1"),
    Sample("meg", "MEG Elekta sample dataset", "XQoEqsa42bNQ4JX", 475, "1"),
    Sample("multimodal", "Multimodal: MRI, MRS, PET, EEG, MEG, eye tracking",
           "e9oZzrJbfBtiaTQ", 125, "3"),
    Sample("multimodal2", "Multimodal 2: one participant over four visits",
           "YaYyq5RzmoY7WXJ", 125, "1"),
)


def find(key: str) -> Sample:
    """The sample called ``key``; ``KeyError`` names the ones there are."""
    for sample in SAMPLES:
        if sample.key == key:
            return sample
    raise KeyError(f"no sample called {key!r}; there are "
                   f"{', '.join(s.key for s in SAMPLES)}")


__all__ = ["SAMPLES", "SAMPLE_DATA_PAGE", "Sample", "find"]
