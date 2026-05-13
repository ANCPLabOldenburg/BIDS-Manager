"""QThread bridges between core logic and the GUI.

Reference: architecture.md §12.

Rule: workers import core modules (``cli/``, ``inventory/``, etc.) but
never import widgets. The GUI subscribes to worker signals and updates
its model on the main thread. The GUI thread therefore never blocks on
core operations.

Public surface:

* :class:`ScanWorker` — runs :func:`bidsmgr.cli.scan.run_scan` on a
  background thread.

Future workers (one per CLI verb): ``ConvertWorker``,
``MetadataWorker``, ``ValidateWorker``.
"""

from .convert import ConvertWorker
from .file_report import FileReportWorker, FolderReportWorker
from .metadata import MetadataWorker
from .nifti_loader import NiftiLoaderWorker
from .report import ReportWorker
from .scan import ScanWorker
from .validate import ValidateWorker

__all__ = [
    "ConvertWorker",
    "FileReportWorker",
    "FolderReportWorker",
    "MetadataWorker",
    "NiftiLoaderWorker",
    "ReportWorker",
    "ScanWorker",
    "ValidateWorker",
]
