"""Calling pet2bids politely.

BIDS Manager uses `pypet2bids <https://github.com/openneuropet/PET2BIDS>`_ for
the parts of PET it solved first and solved well: reading ECAT7 headers, and
turning PMOD blood exports into BIDS tables. Both are the reference
implementation from the group that wrote that part of the standard.

Calling a library written to be a command-line tool takes two courtesies, and
they are the same courtesies wherever we call it, so they live here rather than
being repeated at each site.

**Telemetry off.** The modules we call contain none, but sibling modules in the
package report usage, and a future release could move it. Our user chose BIDS
Manager; they cannot consent to reporting from a package they did not know they
installed. A user who wants to support the project can turn it on deliberately.

**Its console quietened.** A CLI is right to narrate. Inside another
application the same lines are noise at best and misleading at worst: it reports
a missing session id because we hand it a scratch directory, and it warns that
fields are missing which we ask for in the template and write later.
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager

log = logging.getLogger(__name__)

_LOGGER_NAME = "pypet2bids"


def telemetry_off() -> None:
    """Switch off usage reporting before anything from the package is imported.

    ``setdefault`` rather than an assignment: someone who deliberately opted in
    keeps their choice.
    """
    os.environ.setdefault("PET2BIDS_TELEMETRY_ENABLED", "0")


@contextmanager
def quiet_pet2bids():
    """Keep pet2bids' own console output out of the conversion log.

    Their logger is built lazily by a factory that sets DEBUG on it as it goes,
    so quietening it beforehand does nothing: the factory has not run yet, and
    when it does it overwrites the level. Calling the factory here materialises
    and caches the logger, after which the level sticks.
    """
    telemetry_off()
    try:
        from pypet2bids import helper_functions

        logger = helper_functions.logger(_LOGGER_NAME)
    except Exception:  # noqa: BLE001 - their internals are not a contract
        logger = logging.getLogger(_LOGGER_NAME)

    was = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="pypet2bids.*",
            )
            yield
    finally:
        logger.setLevel(was)


__all__ = ["quiet_pet2bids", "telemetry_off"]
