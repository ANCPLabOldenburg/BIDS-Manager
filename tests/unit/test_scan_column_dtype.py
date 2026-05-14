"""Regression tests for the pandas-3 strict-StringDtype fallout.

Newer pandas (>= 2.3, the default on Python 3.14) infers a
``StringDtype`` from a bare ``df[col] = ""`` assignment. That column
then refuses subsequent ``df.at[idx, col] = <float or bool>`` calls
with a ``TypeError``. ``bidsmgr.cli.scan._init_object_column`` exists
to keep these columns object-dtyped so they accept mixed types.

These tests assert the property directly so the bug cannot resurface.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bidsmgr.cli.scan import (
    BIDS_GUESS_COLUMNS,
    PROBE_COLUMNS,
    _init_object_column,
)


def test_helper_initialises_object_dtype() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    _init_object_column(df, "x")
    assert df["x"].dtype == object
    assert list(df["x"]) == ["", "", ""]


def test_object_column_accepts_float_via_at() -> None:
    """The regression: float assignment via ``df.at`` after init."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    _init_object_column(df, "bids_guess_confidence")
    df.at[1, "bids_guess_confidence"] = 0.75
    assert df.at[1, "bids_guess_confidence"] == 0.75


def test_object_column_accepts_bool_via_at() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    _init_object_column(df, "bids_guess_skip")
    df.at[2, "bids_guess_skip"] = True
    assert df.at[2, "bids_guess_skip"] is True


def test_object_column_accepts_strings_too() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    _init_object_column(df, "bids_guess_classifier")
    df.at[0, "bids_guess_classifier"] = "dcm2niix_bidsguess"
    assert df.at[0, "bids_guess_classifier"] == "dcm2niix_bidsguess"


@pytest.mark.parametrize("col", BIDS_GUESS_COLUMNS)
def test_every_bids_guess_column_accepts_mixed_types(col: str) -> None:
    """Each BIDS_GUESS column must accept str, float, bool, None after init.

    The actual column-by-column dtype is intentionally object so the
    augmentation pass in ``_augment_dataframe`` does not have to think
    about which column expects which type.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    _init_object_column(df, col)
    df.at[0, col] = "string-value"
    df.at[1, col] = 0.5
    df.at[2, col] = True
    assert df.at[0, col] == "string-value"
    assert df.at[1, col] == 0.5
    assert df.at[2, col] is True


@pytest.mark.parametrize("col", PROBE_COLUMNS)
def test_every_probe_column_accepts_mixed_types(col: str) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    _init_object_column(df, col)
    df.at[0, col] = "ext.nii.gz"
    df.at[1, col] = 7
    df.at[2, col] = 3.14


def test_strict_string_inference_does_not_break_helper() -> None:
    """Simulate the Python 3.14 / pandas-3 strict-string-dtype default.

    When ``pd.options.future.infer_string`` is on, plain
    ``df[col] = ""`` produces a ``StringDtype`` column that rejects
    float assignments. The helper must keep producing ``object`` dtype
    even in that environment so the augmentation pass in
    ``_augment_dataframe`` keeps working.
    """
    original = pd.get_option("future.infer_string")
    pd.set_option("future.infer_string", True)
    try:
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Sanity check the strict mode is actually active. If a bare
        # init produced object dtype we'd be testing nothing.
        df["confirm_strict"] = ""
        if df["confirm_strict"].dtype == object:
            pytest.skip(
                "pandas version too old to infer strict string dtype; "
                "regression cannot trigger here."
            )

        # The fix: helper must not pick up the strict dtype.
        _init_object_column(df, "bids_guess_confidence")
        assert df["bids_guess_confidence"].dtype == object
        df.at[0, "bids_guess_confidence"] = 0.5
        assert df.at[0, "bids_guess_confidence"] == 0.5
    finally:
        pd.set_option("future.infer_string", original)
