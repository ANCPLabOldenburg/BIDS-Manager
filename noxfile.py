"""The test matrix: every tier, on every supported Python.

Run it anywhere. That is the whole point: this file is the ONE definition of
"run the tests", so the Mac, the Windows box, the Linux box and any CI all do
the identical thing instead of four shell invocations that drift apart.

    nox                     # every session on every Python
    nox -s unit             # one tier, every Python
    nox -s unit-3.12        # one tier, one Python
    nox -l                  # list what there is

Interpreters come from ``uv``, which downloads any it does not have. There is
no pyenv, no conda and no manual install step: a machine needs uv and a git
checkout, and the matrix works. Measured on this Mac, a cold 3.12 environment
builds and runs the 1,417 unit tests in about 40 seconds.

Why nox and not tox: the matrix has conditional shape (the GUI tier needs an
offscreen platform, the real-data tier needs env vars and is skipped when they
are absent), and expressing that in Python is honest where expressing it in
ini syntax is a puzzle.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nox

# uv provides the interpreters and resolves far faster than pip.
nox.options.default_venv_backend = "uv"
# A failing session should not stop the rest of the matrix: the point of a
# matrix is to learn which cells fail, not the first one.
nox.options.stop_on_first_error = False
# Bare `nox` runs what a push runs: everything except the slow GUI row and
# the real-data tier. `nox -s everything` adds the rest.
nox.options.sessions = ["lint", "unit", "installed", "integration", "wheel"]

# The versions pyproject.toml claims to support. Keep them in step.
PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

HERE = Path(__file__).parent


def _install(session: nox.Session, *extras: str) -> None:
    """The package plus what this session needs, from the local checkout."""
    session.install("-e", ".")
    if extras:
        session.install(*extras)


def _targets(session: nox.Session, default: str) -> list[str]:
    """What to run and with which flags, from whatever was passed after ``--``.

    ``session.posargs`` used to be appended after the default directory, which
    is right for a flag and wrong for a path: ``-- tests/unit/test_remove.py``
    became ``pytest tests/unit tests/unit/test_remove.py`` and ran the whole
    directory as well as the file. Naming a file is the most ordinary thing
    somebody wants from a test runner, so a path REPLACES the default and
    anything else is added to it.

    A path is recognised by existing on disk, or by carrying ``::``, which is
    how pytest names a single test inside a file.

        nox -s unit-3.12                                  # the whole tier
        nox -s unit-3.12 -- -k companions                 # filtered
        nox -s unit-3.12 -- tests/unit/test_remove.py     # one file
        nox -s gui-3.10  -- tests/gui/test_converter_panel.py::test_x
    """
    paths, flags = [], []
    for arg in session.posargs:
        if "::" in arg or (HERE / arg).exists() or Path(arg).exists():
            paths.append(arg)
        else:
            flags.append(arg)
    return (paths or [default]) + flags


@nox.session(python=PYTHONS)
def unit(session: nox.Session) -> None:
    """Engine and CLI unit tests. No display, no data, no network."""
    _install(session, "pytest")
    # -v: every test is named as it runs. A progress bar tells you how far
    # along a failing run got and nothing about what it was doing, and the
    # point of a CI log is to be readable after the fact by somebody who was
    # not watching. -ra then repeats every skip and xfail WITH ITS REASON at
    # the end, so "why was that not run" is answerable without scrolling.
    session.run(
        "pytest", *_targets(session, "tests/unit"), "-v", "-ra",
        "--durations=10",
    )


@nox.session(python=PYTHONS)
def gui(session: nox.Session) -> None:
    """Qt tests, headless.

    ``offscreen`` is what makes this runnable on a server, in a container and
    over SSH with no display attached. It is set here rather than being left
    to the caller so that forgetting it cannot silently change what is tested.
    """
    _install(session, "pytest", "pytest-qt")
    session.env["QT_QPA_PLATFORM"] = "offscreen"

    # Every mutable cache Qt touches gets its own directory, per session.
    #
    # Several matrix cells run at once on one laptop, each starting offscreen
    # Qt as the same user, and by default they share ~/.cache/fontconfig and
    # $XDG_RUNTIME_DIR. Concurrent fontconfig cache builds are a known crash,
    # and two GUI cells segfaulted inside Qt on the Linux runner while the
    # same suite passes when run by hand, where there is only ever one copy.
    #
    # Isolating costs a few megabytes per session and removes the sharing.
    private = Path(session.create_tmp())
    for var in ("XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME",
                "XDG_DATA_HOME"):
        home = private / var.lower()
        home.mkdir(parents=True, exist_ok=True)
        # XDG_RUNTIME_DIR is required to be 0700 and Qt complains otherwise.
        home.chmod(0o700)
        session.env[var] = str(home)

    session.run(
        "pytest", *_targets(session, "tests/gui"), "-v", "-ra",
        "--durations=10", "-p", "no:randomly",
    )


@nox.session(python=PYTHONS)
def integration(session: nox.Session) -> None:
    """End-to-end CLI runs against the published sample data.

    scan -> convert -> metadata -> validate on one participant with MRI, PET,
    EEG and MEG, plus a real two-worker joblib run. This is the tier the
    recent defects belonged to, and the one that most needs all three
    platforms: its failures are about paths, processes and exit codes.

    VERBOSE, unlike the other tiers. There are a couple of dozen tests here
    and each one is a named behaviour ("the scans table lists every
    recording"), so a CI log that prints them is a readable account of what
    was checked. The same flag on 1,417 unit tests would be a wall.
    """
    _install(session, "pytest")
    session.env["QT_QPA_PLATFORM"] = "offscreen"
    session.run(
        "pytest", *_targets(session, "tests/integration"), "-v", "-ra",
        "--durations=10",
    )


@nox.session(python=PYTHONS[-1])
def lint(session: nox.Session) -> None:
    """Ruff on the code, actionlint on the workflows. One interpreter: neither
    varies by Python.

    actionlint is here because GitHub validates a workflow only when it tries
    to RUN it, and rejects the whole file for one bad expression. The first
    version of ci.yml used ``${{ runner.temp }}`` in a workflow-level ``env``,
    where the ``runner`` context does not exist, and the only way to find out
    was to push it and watch the run refuse to start. That is a slow way to
    learn a typo.
    """
    # Ruff is PINNED. Unpinned, this job goes red the day ruff ships a
    # release that enables a rule by default, which is a linter release and
    # not a change to this code. Measured: 0.15.22 reports 82 findings on
    # bidsmgr, 0.16.7 reports 1,418, almost all of them UP045 and I001.
    session.install("ruff==0.15.22", "actionlint-py")

    # actionlint FIRST. nox stops a session at the first failing command, so
    # running ruff first would mean a style finding hides a broken workflow,
    # and a broken workflow is the more expensive of the two: GitHub rejects
    # the whole file and nothing runs at all.
    workflows = HERE / ".github" / "workflows"
    if workflows.is_dir():
        session.run(
            "actionlint", *[str(p) for p in sorted(workflows.glob("*.yml"))],
        )

    # bidsmgr only, and bidsmgr/vendor excluded in pyproject: the tests are
    # not linted today, and widening the target is a separate decision from
    # getting a matrix running.
    session.run("ruff", "check", "bidsmgr", *session.posargs)


@nox.session(python=PYTHONS[0])
def real_data(session: nox.Session) -> None:
    """The tier that needs your datasets, so it never runs in hosted CI.

    Skips itself rather than failing when the data is not there, because a
    machine without it is not a machine with a problem. ``BIDSMGR_TEST_DATA``
    points at the raw-data root; the per-modality gates stay as they are.
    """
    root = os.environ.get("BIDSMGR_TEST_DATA")
    if not root or not Path(root).is_dir():
        session.skip(
            "set BIDSMGR_TEST_DATA to the raw-data root to run this tier"
        )
    _install(session, "pytest", "pytest-qt")
    session.env["QT_QPA_PLATFORM"] = "offscreen"
    session.run("pytest", *_targets(session, "tests/real_data"), "-q")


@nox.session(python=PYTHONS[-1])
def wheel(session: nox.Session) -> None:
    """Build a wheel and prove it carries what it must.

    This exists because a correct source tree can still build a broken wheel:
    package data is opt-in, and the vendored bidsval schemas were missing from
    one for exactly that reason. Checking the built artefact rather than the
    checkout is the only way to see it.
    """
    session.install("build")
    out = HERE / "dist" / "noxcheck"
    shutil.rmtree(out, ignore_errors=True)
    session.run("python", "-m", "build", "--wheel", "--outdir", str(out))
    session.run("python", str(HERE / "tools" / "check_wheel.py"), str(out))


@nox.session(python=PYTHONS)
def installed(session: nox.Session) -> None:
    """Install the WHEEL into a clean environment and use the tool.

    The tier every other one is blind to.

    ``pyproject.toml`` sets ``pythonpath = ["."]``, so pytest puts the source
    tree first on ``sys.path``: the unit, gui and integration sessions install
    the package and then import ``./bidsmgr`` regardless. That is correct for
    developing, and it means the suite can never see a packaging defect. The
    vendored bidsval schemas were missing from every wheel for weeks while the
    tree was complete and every test was green.

    So this one installs the built artefact, and runs from a temporary
    directory OUTSIDE the checkout so nothing can shadow it. It goes from
    "pip installed it" to "a dataset converted with no errors", which is the
    whole distance a user travels.
    """
    session.install("build")
    out = HERE / "dist" / f"installed-{session.python}"
    shutil.rmtree(out, ignore_errors=True)
    session.run("python", "-m", "build", "--wheel", "--outdir", str(out))

    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        session.error("the build produced no wheel")
    # --force-reinstall so a cached copy of the same version cannot stand in
    # for the wheel just built.
    session.install("--force-reinstall", str(wheels[-1]))

    try:
        # Somewhere the checkout is not importable. Running this from HERE
        # would put ./bidsmgr on sys.path via the current directory and
        # quietly test the source tree again, which is the exact mistake this
        # session exists for.
        elsewhere = Path(session.create_tmp())
        session.chdir(elsewhere)
        session.run("python", str(HERE / "tools" / "check_installed.py"))
    finally:
        # The wheel has done its job. Five of these, one per interpreter, is
        # a few hundred megabytes of identical artefacts nobody looks at.
        shutil.rmtree(out, ignore_errors=True)


@nox.session(python=False)
def clean(session: nox.Session) -> None:
    """Delete what the matrix leaves behind, and say how much it was.

    Nothing here is cleaned automatically, and the numbers are not small: an
    environment carrying PyQt6, mne, nibabel and pandas is about 570 MB, and
    the full matrix builds twenty of them. That is roughly 11 GB per machine,
    which matters on a laptop.

    CI does not need this. ``actions/checkout`` runs ``git clean -ffdx``,
    which removes ignored files too, so a runner starts every job with an
    empty workspace whether it wants to or not.

    The downloaded sample data is NOT touched: it lives outside the checkout
    on purpose, it is expensive to fetch, and it is shared between runs.
    ``nox -s clean -- data`` removes that too.
    """
    targets = [HERE / ".nox", HERE / "dist", HERE / "build",
               HERE / ".pytest_cache"]
    if "data" in session.posargs:
        from tests.fixtures.sample_data import cache_root

        targets.append(cache_root())

    total = 0
    for target in targets:
        if not target.exists():
            continue
        size = sum(
            p.stat().st_size for p in target.rglob("*") if p.is_file()
        )
        total += size
        session.log(f"{size / 1e9:6.2f} GB  {target}")
        # .nox holds the environment this session is running from, so on some
        # platforms it cannot delete itself. Report rather than pretend.
        shutil.rmtree(target, ignore_errors=True)
    session.log(f"{total / 1e9:6.2f} GB freed")


@nox.session(python=False)
def everything(session: nox.Session) -> None:
    """The whole matrix, exactly what a nightly CI run does.

    A convenience so "run what CI runs" is one command rather than a list to
    remember. ``python=False`` because this session runs no code itself; it
    only chains the others, each of which builds its own environment.

    Roughly 40 minutes on one machine, most of it the GUI row at five minutes
    a cell. CI splits it across two laptops running in parallel.
    """
    names = (
        ["lint"]
        + [f"unit-{v}" for v in PYTHONS]
        + [f"installed-{v}" for v in PYTHONS]
        + [f"gui-{v}" for v in PYTHONS]
        + [f"integration-{v}" for v in PYTHONS]
        + ["wheel"]
    )
    session.log(f"running {len(names)} sessions; see `nox -l` for the list")
    # Re-entrant rather than recursive: each named session gets its own
    # environment, and a failure in one does not stop the rest, which is the
    # same contract the CI matrix has.
    session.run(
        "nox", "-s", *names, "--no-stop-on-first-error", external=True,
    )
