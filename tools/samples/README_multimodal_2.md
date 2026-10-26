# BIDS Manager: multimodal sample 2 (version 1)

One participant, recorded with MRI, MR spectroscopy, PET, EEG, MEG and eye
tracking over four visits, the way a multimodal study records one person:
every file says who it is in the field its format has for it, and every file
says when it was made, on the day of its visit.

## What is in here

```
raw/sub-001/
├── mri/          Siemens Prisma DICOM, visit 1 (14 Sep 2023, 10:13 to 10:41):
│                   T1w (MPRAGE) and a SPACE inversion-recovery T2 (FLAIR)
│                   a resting-state BOLD (60 volumes) with its SBRef
│                   the physiological log recorded during that BOLD run
│                   a phase-difference fieldmap (two magnitudes and a phase)
│                   a diffusion run (AP, 24 volumes) and its reversed-phase
│                   reference (PA, one volume)
│                   three scout reconstructions and a PhoenixZIPReport,
│                   which the scan leaves out on its own
├── mrs/          Siemens single-voxel spectroscopy (HERMES) and its water
│                 reference, the last two series of visit 1 (10:42, 10:50)
├── eyetracking/  one binocular SR Research EyeLink recording, visit 1, started
│                 at 10:19:30 and recording through the resting BOLD run
│                 (10:20:12)
├── eeg/          two EDF+ recordings, 64 channels at 160 Hz, visit 2
│   ├── task-rest_eeg.edf              (21 Sep 2023, 14:00)
│   └── task-motorimagery_eeg.edf      (21 Sep 2023, 14:02)
├── meg/          one Elekta Neuromag (TRIUX) recording, the first 30 s, visit 3
│   └── task-rest_meg.fif              (28 Sep 2023, 09:30)
└── pet/          GE Advance DICOM, one FDG scan (35 files), visit 4, at
                  another site (5 Oct 2023, injection 09:23, scan 09:28)

participants.csv  the participant's demographics, as the headers state them
dose_sheet.csv    the PET fields the scanner did not record (injected mass,
                  specific activity, mode of administration, ...)
```

About 311 MB unpacked, 125 MB as a zip.

## One participant

The study code is `OL_3844`: male, born 1993 (30 at the MRI), 66 kg, 1.73 m.

| Recording | Where it says who | When |
|---|---|---|
| MRI, spectroscopy | DICOM patient module, one study | 14 Sep 2023 |
| Eye tracking | its folder (an EyeLink file names no participant) | 14 Sep 2023, during the BOLD run |
| EEG | EDF+ patient field: `OL_3844 M 01-JAN-1993 X` | 21 Sep 2023 |
| MEG | FIF subject: `OL_3844`, male, right-handed, born 1993 | 28 Sep 2023 |
| PET | DICOM patient module, the PET site's own study | 5 Oct 2023 |

## What it shows

Scan `raw/` and the inventory has 19 rows: 15 to convert and 4 the scan leaves
out on its own (three scout reconstructions and the report). Every recording
is proposed as one subject, `sub-001`: the DICOM series by the patient fields
they share, EEG and MEG by their folder, and the eye tracker, whose file names
no participant, by its folder too, because everything else in it is `sub-001`.

What is left to decide in the table is what a real study leaves to decide:

- **The eye tracker's run.** It is proposed with no session, in `beh/`. In its
  **Properties**, under Eye tracking, choose the **Run** it was recorded
  during, the resting BOLD run: it takes that run's session and task, and
  moves beside it in `func/`.
- **The session labels.** The MRI operator named the series `ses-pre_...`, so
  the MRI rows are proposed as `ses-pre`; the spectroscopy and the PET, whose
  names say nothing, are numbered from their DICOM studies (`ses-1`, `ses-2`),
  and EEG and MEG are numbered on their own (`ses-1`, `ses-2`), so the same
  label is proposed for different days. Give each visit one label, for example
  `ses-mri`, `ses-eeg`, `ses-meg` and `ses-pet`: one edit per column.

Then, as in the multimodal sample:

- **Fieldmaps linked to the runs they correct** (`IntendedFor`,
  `B0FieldIdentifier` / `B0FieldSource`).
- **Physiological recordings, one table per sampling rate**, the ECG leads,
  the pulse and the breathing, each with the scanner trigger, which marks the
  run's 60 volumes.
- **Spectroscopy as NIfTI-MRS**, in `mrs/`.
- **PET with what the scanner did not record.** Import `dose_sheet.csv` in the
  dataset metadata (PET dose file, Import): the injected mass (0.0458 ug of
  FDG), the specific activity (1.66e15 Bq/g, the injected 75.85 MBq the PET
  header records over that mass) and the mode of administration (bolus).
- **Eye tracking as BIDS physio**, one pair of files per eye.

Converted and validated with BIDS Manager 1.5.2 on 2026-10-26, the dose sheet
imported and the scan's proposals accepted as they are: 66 files and four
errors, two PET reconstruction fields no scanner records and the eye tracker's
`StimulusPresentation` (one per eye); the warnings are the TODO placeholders
for what no file records (authors, licence, task descriptions).

## This is a composite, made coherent

The recordings come from different real acquisitions. Here they were given
one participant and one timeline, so that the sample behaves like a single
study. The PET scan is of a phantom, not of a person. What was changed, and
nothing else:

- **Spectroscopy**: recorded in another session, moved into the MRI's study
  (study UID, frame of reference, the patient fields), numbered as series 27
  and 28 after the last MRI series, its times moved to just after it.
- **PET**: the patient fields; every date in the file moved by the same
  number of whole days to 5 Oct 2023 (times of day kept); the study and series
  descriptions no longer name the phantom (`BRAIN^FDG`, `2d_brain_lt_ramp`).
- **EEG**: the EDF+ header's patient and recording fields, start date and
  start time.
- **MEG**: the measurement date and the subject, written back by MNE in the
  file's own format.
- **Eye tracking**: the recording date in the EyeLink file's preamble.
- **The BOLD run's physiological log**: cut to the run's 60 volumes, as in
  the multimodal sample version 3.
- `participants.csv` (age 30, as the MRI headers state) and `dose_sheet.csv`
  (realistic FDG values, consistent with the injected activity).

Every image, signal and spectrum is as it was recorded. The 4-D MRI series
were shortened to keep the download small (the BOLD run to its first 60 of
100 volumes, the diffusion run to its first 24 of 76, its reference to the
first of 5) and the MEG recording to its first 30 seconds.

The sample is built by `tools/samples/build_multimodal.py` in the BIDS Manager
repository, from the multimodal sample version 2.

## Where this came from, and what you may do with it

Each modality keeps the licence of the data it was taken from.

**MRI and MR spectroscopy**: the Siemens sample data of the neuroimaging unit
at the University of Oldenburg. The participant whose anatomical images are
included consented to their face being shown.

**PET**: one subject from the OpenNeuro PET brain phantoms, a GE Advance FDG
acquisition from the NIH Clinical Center. Source: PublicnEUro `PN000001`,
OpenNeuro PET Brain phantoms V2, DOI
[10.70883/RGN3768](https://doi.org/10.70883/RGN3768). Licence: CC BY 4.0, see
`LICENCE-pet-images-CC-BY-4.0.txt`. Changed as listed above.

**EEG**: two runs from subject S001 of the PhysioNet EEG Motor Movement and
Imagery Database, renamed from `S001R01.edf` and `S001R03.edf` so the task is
legible. Licence: Open Data Commons Attribution Licence v1.0, see
`LICENCE-eeg-ODC-By-1.0.txt`. Source: <https://physionet.org/content/eegmmidb/>.
Changed as listed above.

**MEG**: one Elekta Neuromag recording from the BIDS Manager MEG sample.

**Eye tracking**: the binocular test recording shipped with eyelinkio, renamed
from `test_raw_binocular.edf`. Licence: BSD 3-Clause, see
`LICENCE-eyetracking-BSD-3.txt`. Source: <https://github.com/scott-huberty/eyelinkio>.
Changed as listed above.

`participants.csv` and `dose_sheet.csv` were written for this sample by the
BIDS Manager maintainers. Use them however you like.
