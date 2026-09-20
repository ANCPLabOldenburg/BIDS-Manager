"""Every setting must come back out of QSettings as what went in.

This exists because one string setting was written by the loop that saves
BOOLEANS, which stores ``"1"`` for anything truthy. ``convert_deface_engine``
went in as ``"allineate"`` and came back as ``"1"``, the converter could not
resolve that id, and the lookup raised INSIDE the subject commit, so every
conversion produced nothing at all. Defacing worked in the Editor throughout,
because that dialog chooses its own engine and never reads the setting, which
is exactly why it took a real conversion to notice.

So the test is generic: save, reload, compare the whole dataclass. A string
added to the wrong list fails here rather than in somebody's conversion.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402

from bidsmgr.gui.app_settings import KEYS, AppSettings  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Write to a throwaway .ini so the developer's real settings survive."""
    path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        AppSettings, "_settings",
        staticmethod(lambda: QSettings(str(path), QSettings.Format.IniFormat)),
        raising=False,
    )
    yield path


def _round_trip(settings: AppSettings) -> AppSettings:
    settings.save()
    return AppSettings.load()


def test_defaults_survive_a_round_trip():
    before = AppSettings()
    after = _round_trip(before)
    assert dataclasses.asdict(after) == dataclasses.asdict(before)


def test_the_deface_engine_is_not_flattened_to_a_bool():
    """The regression. It was saved by the bool loop and came back as "1"."""
    s = AppSettings()
    s.convert_deface = True
    s.convert_deface_engine = "allineate-robust"

    after = _round_trip(s)

    assert after.convert_deface is True
    assert after.convert_deface_engine == "allineate-robust", (
        "the engine id was flattened; every conversion would fail on it"
    )


def test_every_engine_id_survives():
    from bidsmgr.deface.engines import engine_ids

    for engine_id in engine_ids():
        s = AppSettings()
        s.convert_deface_engine = engine_id
        assert _round_trip(s).convert_deface_engine == engine_id


def test_a_stale_engine_id_heals_instead_of_breaking_the_conversion():
    """Settings written by an older build can hold anything.

    Loading has to produce something the converter can resolve, because the
    alternative is what actually happened: an unresolvable id raising in the
    middle of committing a subject.
    """
    from bidsmgr.deface.engines import engine_ids

    s = AppSettings()
    s.save()
    qs = AppSettings._settings()
    qs.setValue(KEYS["convert_deface_engine"], "1")
    qs.sync()

    loaded = AppSettings.load()
    assert loaded.convert_deface_engine in engine_ids()


def test_every_string_setting_round_trips_a_non_boolish_value():
    """Generic guard: any string field put in the bool loop fails here."""
    # A field whose load path validates against a vocabulary needs a value
    # from that vocabulary, or the round trip legitimately rejects the probe
    # and the test measures the guard instead of the save.
    constrained = {"nifti_view_mode": "combo"}
    probes = {
        str: "allineate-robust",
    }
    fields = [
        f for f in dataclasses.fields(AppSettings)
        if f.type in (str, "str") and f.name in KEYS
    ]
    assert fields, "no string settings found; the introspection broke"

    for field in fields:
        s = AppSettings()
        current = getattr(s, field.name)
        # Use the field's own default when it is a constrained vocabulary;
        # a free-form field gets a value no bool encoder could produce.
        probe = constrained.get(field.name) or current or probes[str]
        setattr(s, field.name, probe)
        after = _round_trip(s)
        assert getattr(after, field.name) == probe, (
            f"{field.name} did not survive; it is probably saved by the "
            "loop that writes booleans as '1'/'0'"
        )
