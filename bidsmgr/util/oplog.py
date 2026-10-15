"""One log file per operation: a scan, a conversion, a metadata pass, a
validation.

Before this nothing was written anywhere: the CLI logged to the terminal and
the GUI's Log panel was a rolling buffer that was never saved, so "what
happened to that participant last Tuesday" had no answer. Each operation now
leaves ``<dataset>/.bidsmgr/logs/<UTC stamp>_<verb>.log`` (outside a project:
``~/.bidsmgr/logs/``): a header saying what ran, where and with what, every
record from INFO up of BIDS Manager and the libraries it drives, and a footer
with the outcome, the counts of warnings and errors, and the duration.

The terminal is not made louder: console handlers keep the level the user
asked for while the file takes INFO and up.

Never log a header's identity values (names, birth dates): the file lives
inside the dataset's tool folder (``.bidsmgr/``, already in ``.bidsignore``),
but it is still a file people send each other.

Qt-free.
"""

from __future__ import annotations

import functools
import inspect
import logging
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional

#: Where a dataset keeps its logs, relative to its root.
LOG_DIR = Path(".bidsmgr") / "logs"
#: How many log files a folder keeps; the oldest go first.
KEEP = 200
#: The loggers whose records the file takes, besides ``bidsmgr``.
LIBRARIES = ("mne", "mne_bids", "nibabel", "pydicom")


def dataset_root_for(path: Optional[Path]) -> Optional[Path]:
    """The dataset ``path`` belongs to: the nearest folder (itself or a
    parent) holding a ``.bidsmgr`` folder, else ``None``."""
    if path is None:
        return None
    p = Path(path)
    for candidate in (p, *p.parents):
        if (candidate / ".bidsmgr").is_dir():
            return candidate
    return None


def log_dir(root: Optional[Path]) -> Path:
    """The folder a dataset's (or, without one, the user's) logs go to."""
    return Path(root) / LOG_DIR if root is not None else Path.home() / ".bidsmgr" / "logs"


def stamp() -> str:
    """A UTC time stamp safe in a file name on every OS (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-4] + "Z"


def latest(root: Optional[Path], verb: Optional[str] = None) -> Optional[Path]:
    """The newest log in ``root``'s log folder (of ``verb`` when given)."""
    folder = log_dir(root)
    if not folder.is_dir():
        return None
    pattern = f"*_{verb}.log" if verb else "*.log"
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def history(root: Optional[Path]) -> list[Path]:
    """Every log in ``root``'s folder, newest first."""
    folder = log_dir(root)
    return sorted(folder.glob("*.log"), reverse=True) if folder.is_dir() else []


def _prune(folder: Path, keep: int) -> None:
    files = sorted(folder.glob("*.log"))
    for old in files[:-keep] if len(files) > keep else []:
        try:
            old.unlink()
        except OSError:
            pass


class _Counter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.warnings = 0
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.errors += 1
        elif record.levelno >= logging.WARNING:
            self.warnings += 1


def _version() -> str:
    try:
        import bidsmgr

        return str(bidsmgr.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


@contextmanager
def operation_log(verb: str, *, root: Optional[Path] = None,
                  details: Optional[Mapping[str, object]] = None,
                  path: Optional[Path] = None,
                  enabled: bool = True) -> Iterator[Optional[Path]]:
    """Write the records of everything inside to one log file.

    ``verb`` names the operation (``scan``, ``convert``...), ``root`` the
    dataset (``None``: the user's log folder), ``details`` what the header
    should say (inputs, outputs, options), ``path`` an explicit file.
    Yields the file's path (``None`` when ``enabled`` is false or the file
    cannot be created; the operation runs either way).
    """
    if not enabled:
        yield None
        return
    target = Path(path) if path is not None else log_dir(root) / f"{stamp()}_{verb}.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning("no log file for this %s: %s", verb, exc)
        yield None
        return
    handler.setLevel(logging.INFO)
    # The engine's records, not a viewer's computing beside it in the GUI.
    handler.addFilter(lambda record: not record.name.startswith("bidsmgr.gui"))
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    counter = _Counter()
    loggers = [logging.getLogger("bidsmgr"), *(logging.getLogger(n) for n in LIBRARIES)]
    saved_levels = [lg.level for lg in loggers]
    # The console keeps the level asked for: a handler with no level of its
    # own would otherwise print every INFO record now let through.
    root_logger = logging.getLogger()
    console = [(h, h.level) for h in root_logger.handlers if h.level == logging.NOTSET]
    for h, _ in console:
        h.setLevel(root_logger.getEffectiveLevel())
    for lg in loggers:
        if lg.getEffectiveLevel() > logging.INFO:
            lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        lg.addHandler(counter)
    head = logging.getLogger("bidsmgr.oplog")
    head.info("BIDS Manager %s: %s", _version(), verb)
    head.info("platform %s, Python %s", platform.platform(), sys.version.split()[0])
    for key, value in (details or {}).items():
        head.info("%s: %s", key, value)
    started = time.monotonic()
    outcome = "finished"
    try:
        yield target
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        outcome = f"stopped by {type(exc).__name__}: {exc}"
        raise
    finally:
        head.info("%s %s after %.1f s: %d warning(s), %d error(s)", verb, outcome,
                  time.monotonic() - started, counter.warnings, counter.errors)
        for lg, level in zip(loggers, saved_levels):
            lg.removeHandler(handler)
            lg.removeHandler(counter)
            lg.setLevel(level)
        for h, level in console:
            h.setLevel(level)
        handler.close()
        if path is None:
            _prune(target.parent, KEEP)


def add_log_arguments(parser) -> None:
    """``--log-file`` and ``--no-log-file`` for a command line."""
    parser.add_argument(
        "--log-file", type=Path, default=None, metavar="PATH",
        help="Write this run's log here (default: <dataset>/.bidsmgr/logs/, or "
             "~/.bidsmgr/logs/ outside a dataset).")
    parser.add_argument(
        "--no-log-file", action="store_true",
        help="Do not write a log file for this run.")


def log_keywords(args) -> dict:
    """The keywords a logged entry point takes, from parsed arguments."""
    return {"log_file": getattr(args, "log_file", None),
            "write_log": not getattr(args, "no_log_file", False)}


def logged(verb: str, where: Callable[[Mapping[str, object]], tuple]):
    """Decorate an operation's entry point so every call writes its log.

    ``where(arguments)`` returns ``(root, details)`` from the call's bound
    arguments. The decorated function takes two more keywords: ``log_file``
    (write there instead) and ``write_log`` (``False``: no file). When a
    person is watching a terminal, the file's path is printed at the end.
    """
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, log_file: Optional[Path] = None, write_log: bool = True, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            root, details = where(bound.arguments)
            with operation_log(verb, root=root, details=details, path=log_file,
                               enabled=write_log) as written:
                try:
                    return fn(*args, **kwargs)
                finally:
                    if written is not None and getattr(sys.stderr, "isatty", lambda: False)():
                        print(f"Log: {written}", file=sys.stderr)
        return wrapper
    return decorate


__all__ = ["KEEP", "LIBRARIES", "LOG_DIR", "add_log_arguments", "dataset_root_for", "history",
           "latest", "log_dir", "log_keywords", "logged", "operation_log", "stamp"]
