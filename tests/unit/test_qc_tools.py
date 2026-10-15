"""The established tools the quality check calls (``bidsmgr.qc.tools``):
niimath's registration, mindgrab's brain, and a cancelled check stopping
them. Skipped where they cannot run (no niimath wheel, no brainchop)."""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from bidsmgr.qc import tools as T
from bidsmgr.qc.templates import template

needs_tools = pytest.mark.skipif(T.unavailable_reason() is not None,
                                 reason=f"tools unavailable: {T.unavailable_reason()}")


def test_the_numpy_engine_can_be_forced(monkeypatch) -> None:
    monkeypatch.setenv("BIDSMGR_QC_ENGINE", "numpy")
    assert "numpy" in T.unavailable_reason()


def test_a_cancelled_call_is_killed_at_once() -> None:
    t0 = time.monotonic()
    with pytest.raises(T.ToolFailed, match="cancelled"):
        T._call([sys.executable, "-c", "import time; time.sleep(30)"], timeout=60,
                 cancel=lambda: time.monotonic() - t0 > 0.5)
    assert time.monotonic() - t0 < 5


def test_a_call_that_outlives_its_timeout_is_killed() -> None:
    with pytest.raises(T.ToolFailed, match="longer than"):
        T._call([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)


@needs_tools
def test_allineate_of_the_template_to_itself_is_the_identity() -> None:
    m = T.allineate(T.template_file("T1w"), T.template_file("T1w"))
    assert np.allclose(m[:3, :3], np.eye(3), atol=0.02)
    assert np.allclose(m[:3, 3], 0.0, atol=1.0)


@needs_tools
def test_mindgrab_finds_the_templates_brain() -> None:
    tpl = template()
    affine = tpl.affine
    got = T.run_models(T.template_file("T1w"), tpl["T1w"].shape, affine, [T.BRAIN_MODEL])
    brain = got[T.BRAIN_MODEL] > 0
    ref = tpl["brain"] > 0.5
    dice = 2 * np.count_nonzero(brain & ref) / (brain.sum() + ref.sum())
    assert dice > 0.85
