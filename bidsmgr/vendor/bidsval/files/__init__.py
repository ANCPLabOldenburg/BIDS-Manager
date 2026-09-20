"""Access to the files of a dataset.

:class:`~bidsval.files.tree.FileTree` is the validator's view of a dataset on
disk: it indexes the files once and answers the questions the context builder and
rule engine ask (iterate files, find sidecars up the tree, list subjects, test
whether a path exists). Keeping all filesystem access behind this one type makes
the rest of the validator easy to test and, later, to back with other sources.
"""

from __future__ import annotations

from .tree import BIDSFile, FileTree

__all__ = ["FileTree", "BIDSFile"]
