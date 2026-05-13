"""Mock data for the Inspector prototype, matching gui_mockups.html proposal 1.

All values come from the real Siemens Prisma 3T dataset at
~/Development/datasets/BIDS_Manager/raw_data/MRI/neuroimaging_unit_new
(subjects OL_0001/0002/0003) so the prototype is directly comparable
to the HTML mockup.
"""

# -----------------------------------------------------------------------------
# Raw FS tree (col 1 of the Inspector)
# Each tuple: (depth, kind, label, meta, state)
#   depth: 0..N
#   kind:  'folder' | 'series' | 'series-skip' | 'series-active'
#   label: display string
#   meta:  optional small text on the right (e.g. "9 series", "M·P")
#   state: 'expanded' | 'collapsed' | None
# -----------------------------------------------------------------------------
RAW_TREE = [
    (0, 'folder',         'neuroimaging_unit_new',          '',           'expanded'),
    (1, 'folder',         'OL_0001',                         '9 series',  'expanded'),
    (2, 'series-skip',    'localizer_20ch_head-coil',        '',           None),
    (2, 'series',         'ses-pre_run-01_fmap',             'M·P',        None),
    (2, 'series',         'ses-pre_run-02_fmap',             'M·P',        None),
    (2, 'series',         'ses-pre_T1w',                     '',           None),
    (2, 'series',         'ses-pre_task-sparse_bold',        '',           None),
    (2, 'series-active',  'ses-pre_task-rest_bold',          '',           None),
    (2, 'series-skip',    'PhoenixZIPReport',                '',           None),
    (1, 'folder',         'OL_0002',                         '10 series', 'collapsed'),
    (1, 'folder',         'OL_0003',                         '12 series', 'collapsed'),
]

# -----------------------------------------------------------------------------
# Modality / structure filter tree (col 2) with tri-state checkboxes.
# Each tuple: (depth, label, count, state)
#   state: 'checked' | 'unchecked' | 'partial'
# -----------------------------------------------------------------------------
FILTER_TREE = [
    (0, 'Studyname',            31, 'checked'),
    (1, 'sub-001',               8, 'checked'),
    (2, 'ses-pre',               8, 'checked'),
    (3, 'anat',                  1, 'checked'),
    (3, 'func',                  2, 'checked'),
    (3, 'fmap',                  4, 'checked'),
    (3, 'misc',                  1, 'unchecked'),
    (1, 'sub-002',              10, 'partial'),
    (2, 'ses-post',             10, 'partial'),
    (3, 'anat',                  4, 'unchecked'),
    (3, 'func',                  3, 'checked'),
    (3, 'fmap',                  2, 'checked'),
    (3, 'physio',                1, 'checked'),
    (1, 'sub-003',              12, 'checked'),
    (2, 'anat',                  1, 'checked'),
    (2, 'func',                  6, 'checked'),
    (2, 'physio',                2, 'checked'),
]

# Always-exclude patterns shown below the filter tree
ALWAYS_EXCLUDE = [
    '*PhoenixZIPReport',
    'localizer*',
    'AAHead_Scout*',
]

# -----------------------------------------------------------------------------
# Inspection table (col 3) — one row per recording.
# (status, sub, ses, mod, datatype, suffix, task, run, conf, basename, backend, row_state, included)
#   status: 'ok' | 'warn' | 'err' | 'phys' | 'skip'
#   row_state: '' | 'selected' | 'warn' | 'err' | 'skip'
# -----------------------------------------------------------------------------
INVENTORY = [
    ('ok',   '001', 'pre',  'mri', 'anat', 'T1w',        '—',     '—',  '.97',
     'sub-001_ses-pre_T1w',                              'dcm2niix', '',         True),
    ('ok',   '001', 'pre',  'mri', 'func', 'bold',       'rest',  '—',  '.97',
     'sub-001_ses-pre_task-rest_bold',                   'dcm2niix', 'selected', True),
    ('ok',   '001', 'pre',  'mri', 'func', 'bold',       'sparse','—',  '.96',
     'sub-001_ses-pre_task-sparse_bold',                 'dcm2niix', '',         True),
    ('ok',   '001', 'pre',  'mri', 'fmap', 'magnitude1', '—',     '01', '.94',
     'sub-001_ses-pre_run-01_magnitude1',                'dcm2niix', '',         True),
    ('ok',   '001', 'pre',  'mri', 'fmap', 'phasediff',  '—',     '01', '.94',
     'sub-001_ses-pre_run-01_phasediff',                 'dcm2niix', '',         True),
    ('warn', '002', 'post', 'mri', 'func', 'sbref',      'mb',    '—',  '.71',
     'sub-002_ses-post_task-mb_sbref',                   'dcm2niix', 'warn',     True),
    ('ok',   '002', 'post', 'mri', 'func', 'bold',       'mb',    '—',  '.95',
     'sub-002_ses-post_task-mb_bold',                    'dcm2niix', '',         True),
    ('phys', '002', 'post', 'mri', 'func', 'physio',     'mb',    '—',  '.99',
     'sub-002_ses-post_task-mb_physio',                  'bidsphysio','',        True),
    ('err',  '003', '—',    'mri', 'func', 'bold',       'missing','01','.61',
     '— task entity required —',                         'dcm2niix', 'err',      True),
    ('ok',   '003', '—',    'mri', 'anat', 'T2w',        '—',     '—',  '.95',
     'sub-003_acq-space_T2w',                            'dcm2niix', '',         True),
    ('ok',   '003', '—',    'mri', 'func', 'bold',       'dmaging','02','.93',
     'sub-003_task-dmaging_run-02_bold',                 'dcm2niix', '',         True),
    ('skip', '001', 'pre',  'mri', '—',    '—',          '—',     '—',  '—',
     'localizer_20ch_head-coil',                         'skip',     'skip',     False),
]

# -----------------------------------------------------------------------------
# Selected row's properties (col 4 of the Inspector)
# -----------------------------------------------------------------------------
SELECTED_PROPS = {
    'datatype': 'func',
    'suffix': 'bold',
    'entities': [
        # (name, value, required, optional_label)
        ('sub',  '001',  True,  None),
        ('ses',  'pre',  False, None),
        ('task', 'rest', True,  None),
        ('acq',  '',     False, 'opt'),
        ('run',  '',     False, 'opt'),
        ('echo', '',     False, 'opt'),
        ('dir',  '',     False, 'opt'),
        ('part', '',     False, 'opt'),
    ],
    'predicted_path': [
        ('plain',   'Studyname/'),
        ('newline', None),
        ('seg',     'sub-001'),
        ('plain',   '/'),
        ('seg',     'ses-pre'),
        ('plain',   '/'),
        ('dim',     'func'),
        ('plain',   '/'),
        ('newline', None),
        ('seg',     'sub-001'),
        ('plain',   '_'),
        ('seg',     'ses-pre'),
        ('plain',   '_'),
        ('ent',     'task-rest'),
        ('plain',   '_'),
        ('suf',     'bold'),
        ('ext',     '.nii.gz'),
    ],
    'validation': [
        ('ok',   'entity set valid for func/bold'),
        ('ok',   'basename matches schema regex'),
        ('ok',   'no path collision'),
        ('info', 'sidecar will derive: TaskName="rest", RepetitionTime, EchoTime, FlipAngle from DICOM'),
    ],
    'why': [
        ('datatype=func, suffix=bold',
         "dcm2niix BidsGuess: ['func','_task-rest_bold'] · conf 0.97"),
        ('task=rest',
         "SeriesDescription regex: task-(\\w+)"),
        ('ses=pre',
         "SeriesDescription token ses-pre (also confirmed by StudyDate)"),
        ('sub=001',
         "PatientID OL_0001 → auto-numbered (sort=GivenName)"),
    ],
}

# -----------------------------------------------------------------------------
# BIDS preview tree (bottom dock)
# (depth, kind, label, badge)
#   kind: 'dir' | 'nii' | 'json' | 'tsv' | 'other'
# -----------------------------------------------------------------------------
BIDS_PREVIEW = [
    (0, 'dir',  'Studyname/',                                              ''),
    (1, 'dir',  'sub-001/',                                                ''),
    (2, 'dir',  'ses-pre/',                                                ''),
    (3, 'dir',  'anat/',                                                   ''),
    (3, 'nii',  '  └ sub-001_ses-pre_T1w.nii.gz',                          'new'),
    (3, 'json', '  └ sub-001_ses-pre_T1w.json',                            ''),
    (3, 'dir',  'func/',                                                   ''),
    (3, 'nii',  '  └ sub-001_ses-pre_task-rest_bold.nii.gz',               'new'),
    (3, 'json', '  └ sub-001_ses-pre_task-rest_bold.json',                 ''),
    (3, 'nii',  '  └ sub-001_ses-pre_task-sparse_bold.nii.gz',             'new'),
    (3, 'dir',  'fmap/',                                                   ''),
    (3, 'nii',  '  └ sub-001_ses-pre_run-01_magnitude1.nii.gz',            'new'),
    (3, 'nii',  '  └ sub-001_ses-pre_run-01_phasediff.nii.gz',             'new'),
    (2, 'dir',  'sub-002/ses-post/...',                                    ''),
    (2, 'dir',  'sub-003/...',                                             ''),
    (1, 'tsv',  'participants.tsv',                                        ''),
    (1, 'json', 'dataset_description.json',                                ''),
]

# Toolbar status counts
TOOLBAR_STATS = {
    'valid': 19,
    'warn': 2,
    'error': 1,
    'skipped': 5,
}

# Status bar
DATASET_INFO = {
    'schema': 'BIDS 1.10.0',
    'rows': 27,
    'subjects': 3,
    'studies': 1,
    'last_scan': '2 min ago',
    'ready': 19,
    'percent': 70,
}


# =====================================================================
#  EDITOR VIEW DATA
# =====================================================================
# Tuple: (depth, kind, label, badge)
#   kind:  'dir' | 'nii' | 'json' | 'tsv' | 'other'
#   badge: '' | 'ok' | 'warn' | 'err'
EDITOR_BIDS_TREE = [
    (0, 'dir',   'Studyname/',                                                      'warn'),
    (1, 'json',  'dataset_description.json',                                        'ok'),
    (1, 'tsv',   'participants.tsv',                                                'ok'),
    (1, 'other', 'README',                                                          'ok'),
    (1, 'other', 'CHANGES',                                                         'ok'),
    (1, 'dir',   'sub-001/',                                                        'warn'),
    (2, 'dir',   'ses-pre/',                                                        'warn'),
    (3, 'dir',   'anat/',                                                           'ok'),
    (4, 'nii',   'sub-001_ses-pre_T1w.nii.gz',                                      'ok'),
    (4, 'json',  'sub-001_ses-pre_T1w.json',                                        'ok'),  # ACTIVE
    (3, 'dir',   'func/',                                                           'warn'),
    (4, 'nii',   'sub-001_ses-pre_task-rest_bold.nii.gz',                           'ok'),
    (4, 'json',  'sub-001_ses-pre_task-rest_bold.json',                             'warn'),
    (4, 'nii',   'sub-001_ses-pre_task-sparse_bold.nii.gz',                         'ok'),
    (4, 'json',  'sub-001_ses-pre_task-sparse_bold.json',                           'ok'),
    (3, 'dir',   'fmap/',                                                           'ok'),
    (4, 'nii',   'sub-001_ses-pre_run-01_magnitude1.nii.gz',                        'ok'),
    (4, 'json',  'sub-001_ses-pre_run-01_phasediff.json',                           'ok'),
    (1, 'dir',   'sub-002/',                                                        'err'),
    (2, 'dir',   'ses-post/',                                                       'err'),
    (3, 'dir',   'fmap/',                                                           'err'),
    (4, 'json',  'sub-002_ses-post_run-01_phasediff.json',                          'err'),
    (3, 'dir',   'func/',                                                           'warn'),
    (4, 'nii',   'sub-002_ses-post_task-mb_bold.nii.gz',                            'ok'),
    (4, 'json',  'sub-002_ses-post_task-mb_bold.json',                              'ok'),
    (4, 'tsv',   'sub-002_ses-post_task-mb_physio.tsv.gz',                          'warn'),
    (1, 'dir',   'sub-003/',                                                        'ok'),
    (2, 'dir',   'anat/',                                                           'ok'),
    (3, 'nii',   'sub-003_acq-space_T2w.nii.gz',                                    'ok'),
    (3, 'json',  'sub-003_acq-space_T2w.json',                                      'ok'),
    (2, 'dir',   'func/',                                                           'ok'),
    (3, 'nii',   'sub-003_task-dmaging_run-01_bold.nii.gz',                         'ok'),
    (3, 'nii',   'sub-003_task-dmaging_run-02_bold.nii.gz',                         'ok'),
    (1, 'dir',   'derivatives/',                                                    ''),
]

# Open file tabs (above the viewer in the editor's center pane)
EDITOR_OPEN_TABS = [
    ('nii',  'sub-001_ses-pre_T1w.nii.gz',     False),
    ('json', 'sub-001_ses-pre_T1w.json',       True),   # ACTIVE
    ('nii',  'sub-001_..._bold.nii.gz',        False),
]

# Schema-driven sidecar fields shown in the JSON viewer.
# Tuple: (level, key, value, value_kind)
#   level:       'req' | 'rec' | 'opt' | 'dep'
#   value_kind:  'str' | 'num' | 'todo'
EDITOR_SIDECAR = {
    'datatype': 'anat',
    'suffix':   'T1w',
    'fields': [
        ('req', 'Modality',                'MR',                   'str'),
        ('rec', 'MagneticFieldStrength',   '3',                    'num'),
        ('rec', 'Manufacturer',            'Siemens Healthineers', 'str'),
        ('rec', 'ManufacturersModelName',  'MAGNETOM Prisma',      'str'),
        ('rec', 'InstitutionName',         'University of Oldenburg', 'str'),
        ('rec', 'DeviceSerialNumber',      '66080',                'str'),
        ('rec', 'SoftwareVersions',        'syngo MR XA31A',       'str'),
        ('rec', 'BodyPartExamined',        'BRAIN',                'str'),
        ('rec', 'ScanningSequence',        'GR\\IR',               'str'),
        ('rec', 'SequenceVariant',         'SK\\SP\\MP',           'str'),
        ('rec', 'EchoTime',                '0.00226',              'num'),
        ('rec', 'InversionTime',           '0.9',                  'num'),
        ('rec', 'FlipAngle',               '8',                    'num'),
        ('rec', 'RepetitionTime',          '2.3',                  'num'),
        ('opt', 'PhaseEncodingDirection',  'j-',                   'str'),
        ('opt', 'AcquisitionDateTime',     '2025-05-26T07:16:29',  'str'),
        ('dep', 'AcquisitionDuration',     '300',                  'num'),
    ],
    'path': '~/datasets/BIDS_out/Studyname/sub-001/ses-pre/anat/sub-001_ses-pre_T1w.json',
    'summary': '17 fields · 14 filled · 3 deprecated',
}

# Validation panel (right side of editor)
# Section: title, count_text, count_kind ('ok'/'warn'/'err')
# Messages: (severity, rule, body, [fix_button_label])
EDITOR_VALIDATION = [
    {
        'title':      'This file',
        'count':      'all clear',
        'count_kind': 'ok',
        'messages': [
            ('ok',
             'SCHEMA · anat/T1w',
             'All REQUIRED fields present. RECOMMENDED <code>InstitutionAddress</code> missing — non-blocking.',
             None),
        ],
    },
    {
        'title':      'This folder · sub-001/',
        'count':      '1 warning',
        'count_kind': 'warn',
        'messages': [
            ('warn',
             'BIDS-FUNC-002 · func/bold',
             '<code>sub-001_ses-pre_task-rest_bold.json</code> missing <code>SliceTiming</code> (RECOMMENDED).',
             'Auto-fill'),
        ],
    },
    {
        'title':      'Dataset',
        'count':      '2 errors · 7 warnings',
        'count_kind': 'err',
        'messages': [
            ('err',
             'BIDS-FUNC-001 · func/bold',
             '<code>sub-002/ses-post/func/sub-002_ses-post_task-mb_bold.json</code> missing REQUIRED <code>TaskName</code>.',
             'Set "mb"'),
            ('err',
             'BIDS-FMAP-005',
             '<code>sub-002/ses-post/fmap/sub-002_ses-post_run-01_phasediff.json</code> <code>IntendedFor</code> array points to 0 files (must be ≥ 1).',
             'Repopulate'),
            ('warn',
             'BIDS-DATASET-008',
             '<code>participants.tsv</code> missing optional column <code>handedness</code>.',
             None),
        ],
    },
]

# Provenance entries
EDITOR_PROVENANCE = [
    ('RepetitionTime=2.3',  'from DICOM tag (0018,0080)'),
    ('EchoTime=0.00226',    'from DICOM tag (0018,0081)'),
    ('Manufacturer',        'from (0008,0070)'),
]

# Editor toolbar
EDITOR_STATS = {
    'valid': 124,
    'warn':  7,
    'error': 2,
    'last_validated': '30s ago',
}

# Schema legend chips
SCHEMA_LEGEND = [
    ('req', 'REQUIRED'),
    ('rec', 'RECOMMENDED'),
    ('opt', 'OPTIONAL'),
    ('dep', 'DEPRECATED'),
]
