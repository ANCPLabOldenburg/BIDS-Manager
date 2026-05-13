"""Converter backend implementations.

Each backend implements the ``ConverterBackend`` Protocol from
``bidsmgr.converter.registry``: ``name``, ``can_handle(task)``, and
``convert(task, staging_dir) -> ConvertResult``.

Modules:
* ``dcm2niix_direct`` — default MRI backend.

Planned later: ``dcm2bids``, ``heudiconv`` (heudiconv-ancp fork),
``mne_bids`` (EEG/MEG/iEEG), ``bidsphysio``, ``passthrough``.
"""
