"""The guide's tasks and their steps, decided from what each view reports.

What each view offers and which one is the usual next; that a task starts
where the user chose it; that a step moves on by itself only when something
happens after it was shown (never because a fact was already true, which is
how "Check and edit the table" used to jump to Run); busy steps; and the
texts.
"""

from __future__ import annotations

import pytest

from bidsmgr.gui.guide_steps import find_task, should_advance, shown, tasks

_TABLE = {"project": True, "rows": 12, "tables": 1, "row_clicks": 0, "raw_picks": 1}


def _keys(view, facts):
    return [t.key for t in tasks(view, facts)]


def _best(view, facts):
    return [t.key for t in tasks(view, facts) if t.recommended]


# ---------------------------------------------------------------------------
# What each view offers
# ---------------------------------------------------------------------------


def test_home_offers_creating_opening_reopening_and_practising() -> None:
    assert _keys("welcome", {}) == ["create", "open", "sample"]
    assert _best("welcome", {}) == ["create"]
    assert _keys("welcome", {"recents": 2}) == ["create", "open", "recent", "sample"]
    assert _best("welcome", {"recents": 2}) == ["recent"]
    with_project = {"project": True, "project_name": "Motor_Study", "recents": 1}
    assert _keys("welcome", with_project)[0] == "continue"
    assert tasks("welcome", with_project)[0].label == "Continue with Motor_Study"
    assert _best("welcome", with_project) == ["continue"]


def test_the_converter_offers_more_once_there_is_a_table() -> None:
    assert _keys("converter", {}) == ["start"]
    assert _keys("converter", {"project": True}) == ["scan", "settings"]
    assert tasks("converter", {"project": True})[0].label == "Scan raw data"
    keys = _keys("converter", _TABLE)
    assert keys == ["scan", "review", "row_meta", "bulk", "columns", "convert", "settings"]
    assert tasks("converter", _TABLE)[0].label == "Scan new data"
    # Dataset metadata where its button works; the Editor once something exists.
    assert "dataset_meta" in _keys("converter", {**_TABLE, "meta_enabled": True})
    assert "editor" in _keys("converter", {**_TABLE, "subjects": True})


def test_the_converters_usual_next() -> None:
    assert _best("converter", {"project": True}) == ["scan"]
    assert _best("converter", _TABLE) == ["review"]
    assert _best("converter", {**_TABLE, "row_clicks": 3}) == ["convert"]
    assert _best("converter", {**_TABLE, "converted": True}) == ["editor"]


def test_the_editor_offers_its_tools() -> None:
    base = {"root": True, "subjects": True}
    assert _keys("editor", base) == ["validate", "edit", "rename", "sessions", "delete",
                                     "anonymise", "quality", "repair", "history"]
    assert _best("editor", base) == ["validate"]
    found = {**base, "validated": True, "errors": 2, "warnings": 5}
    assert _keys("editor", found)[:2] == ["validate", "fix"]
    assert tasks("editor", found)[0].label == "Validate again"
    assert _best("editor", found) == ["fix"]
    assert "2 errors, 5 warnings" in find_task("editor", "fix", found).detail
    assert _best("editor", {**base, "validated": True}) == ["edit"]


def test_where_there_is_one_path_or_none() -> None:
    assert _keys("editor", {}) == ["open"]
    assert _keys("editor", {"root": True}) == ["convert_first"]
    assert _keys("editor", {"root": True, "subjects": True, "view_only": True}) == []
    assert tasks("nowhere", {}) == []


# ---------------------------------------------------------------------------
# Steps: shown as chosen, moved on by what happens
# ---------------------------------------------------------------------------


def test_a_task_starts_at_its_first_step_whatever_is_already_true() -> None:
    """The defect: "Check and edit the table" jumped to Run because a row had
    already been clicked. A fact already true never moves a step."""
    clicked = {**_TABLE, "row_clicks": 5}
    review = find_task("converter", "review", clicked)
    assert shown(review, 0, clicked).target == "table"
    assert not should_advance(review, 0, clicked, clicked)
    # A click AFTER the step was shown moves it on, to Properties.
    assert should_advance(review, 0, clicked, {**clicked, "row_clicks": 6})
    assert shown(review, 1, clicked).target == "properties"


def test_the_scan_task_follows_the_user() -> None:
    scan = find_task("converter", "scan", _TABLE)
    assert [s.target for s in scan.steps] == ["raw_folder", "scan", "table"]
    assert not should_advance(scan, 0, _TABLE, _TABLE)
    assert should_advance(scan, 0, _TABLE, {**_TABLE, "raw_picks": 2})
    busy = {**_TABLE, "scanning": True}
    step = shown(scan, 1, busy)
    assert step.busy and step.title == "Scanning" and step.key == "scan.scan.busy"
    assert not should_advance(scan, 1, _TABLE, {**busy, "tables": 2}), "not while scanning"
    assert should_advance(scan, 1, _TABLE, {**_TABLE, "tables": 2})
    assert shown(scan, 2, _TABLE).done


def test_home_steps_move_on_with_the_user() -> None:
    create = find_task("welcome", "create", {})
    assert [s.target for s in create.steps] == ["browse", "name", "create"]
    assert should_advance(create, 0, {"browse_clicks": 0}, {"browse_clicks": 1})
    assert not should_advance(create, 1, {"name": True}, {"name": True})
    assert should_advance(create, 1, {"name": False}, {"name": True})
    assert should_advance(create, 2, {"project_root": ""}, {"project_root": "/x/Study"})
    assert not should_advance(create, 2, {"project_root": "/x/A"}, {"project_root": "/x/A"})


def test_bulk_edit_waits_for_several_rows() -> None:
    bulk = find_task("converter", "bulk", _TABLE)
    assert not should_advance(bulk, 0, _TABLE, {**_TABLE, "selected_rows": 1})
    assert should_advance(bulk, 0, _TABLE, {**_TABLE, "selected_rows": 3})
    assert shown(bulk, 1, _TABLE).target == "bulk_edit"


def test_converting_and_validating() -> None:
    convert = find_task("converter", "convert", _TABLE)
    assert shown(convert, 0, {**_TABLE, "converting": True}).busy
    assert should_advance(convert, 0, {"conversions": 0}, {"conversions": 1})
    assert shown(convert, 1, _TABLE).target == "editor"
    base = {"root": True, "subjects": True}
    validate = find_task("editor", "validate", base)
    assert shown(validate, 0, {**base, "validating": True}).busy
    assert should_advance(validate, 0, {"reports": 0}, {"reports": 1})
    assert "3 errors to fix" in shown(validate, 1, {**base, "errors": 3}).text
    assert "no warnings" in shown(validate, 1, base).text


def test_the_heading_names_the_task_and_the_step() -> None:
    create = find_task("welcome", "create", {})
    assert shown(create, 1, {}).heading == "Create a new dataset · Step 2 of 3"
    sample = find_task("welcome", "sample", {})
    assert shown(sample, 0, {}).heading == "Practise on a sample dataset"


def test_keys_are_named_as_the_os_names_them(monkeypatch) -> None:
    from bidsmgr.viz import keynames

    monkeypatch.setattr(keynames, "is_mac", lambda: True)
    monkeypatch.setattr(keynames, "_on_mac", lambda mac: True if mac is None else mac)
    text = shown(find_task("converter", "bulk", _TABLE), 0, _TABLE).text
    assert "⌘" in text and "Ctrl" not in text


@pytest.mark.parametrize("view, facts", [
    ("welcome", {"project": True, "recents": 1}),
    ("converter", {**_TABLE, "meta_enabled": True, "subjects": True}),
    ("editor", {"root": True, "subjects": True, "validated": True, "errors": 1}),
])
def test_every_text_is_plain_and_every_step_names_a_control(view, facts) -> None:
    for task in tasks(view, facts):
        assert task.label and task.detail and task.steps
        assert "—" not in task.label + task.detail
        for i in range(len(task.steps)):
            step = shown(task, i, facts)
            assert step.target and step.title and step.text
            assert "—" not in step.title + step.text
