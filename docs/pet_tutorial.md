# Converting PET with BIDS Manager

A practical walkthrough for someone with PET data on disk and a deadline. It
covers the two source formats BIDS Manager reads, DICOM and ECAT, and the blood
data that makes a PET dataset quantitative.

Everything below was run on the OpenNeuroPET phantom set: 21 DICOM series from
14 scanner models plus 3 ECAT files, and real PMOD blood exports.

---

## 1. Before you start: what to gather

PET is unusual in how much of what BIDS requires is not in the image file. Ten
fields are REQUIRED for a PET sidecar and no scanner writes several of them, so
collect these before you sit down:

| you will be asked for | where it usually lives |
|---|---|
| injected radioactivity and its units | the dose sheet or the hot lab log |
| injected mass and its units | same |
| specific radioactivity and its units | same |
| mode of administration (bolus, infusion, ...) | the protocol |
| time zero, and when the injection started | the console clock and the injection log |
| acquisition mode (list mode, static, ...) | the protocol |

You do not have to type these one at a time. Section 5 covers doing it once for
the whole study, and section 8 covers importing your lab's dose spreadsheet.

If you have blood data, find the PMOD `.bld` exports now: typically one file per
curve, whole blood, plasma, and parent fraction.

---

## 2. Create a project and scan

Open BIDS Manager. On the Welcome tab choose **Create**, give the dataset a
folder and a name, and it opens on the Converter tab.

Press **Scan** and point it at the folder holding your raw PET. Point it at the
TOP of the tree, not at one subject: the scanner walks down, and it works out
subjects, sessions and series for itself.

You do not have to say what format it is. Detection is by content, not by file
extension:

* DICOM is read as DICOM whatever the files are called. Philips writes the SOP
  instance UID as the filename and GE Signa writes `.img`; both are read.
* ECAT is detected by the `MATRIX7x` signature at the start of the file, so a
  `.v` file works, and so does one your site renamed.

A PET/CT study is handled without you doing anything: the PET half converts and
the CT half is excluded with a reason, because BIDS has no CT datatype yet. A
PET/MR study converts both halves in one pass, the MR as ordinary anat and func.

**Leave "probe convert" on.** It runs the converter once per series during the
scan and reads what comes out, which is how the template later knows which
questions it does not need to ask you. It costs a little scan time and saves a
lot of typing.

---

## 3. Read the table before you touch anything

The inspection table is one row per recording. Four columns are worth checking
first:

* **include**: untick anything you do not want. An excluded row is never
  written and never clashes with anything.
* **predicted basename**: the BIDS name this row will get. If a name shows in
  **red**, two recordings resolve to the same name and one would overwrite the
  other. Fix it before converting; conversion refuses while any name is
  duplicated, and it names the files involved. Which entity to change depends on
  what actually differs: a task label for different tasks, a session for
  different days, a run for genuine repeats.
* **tracer**, **radionuclide**, **recon method**: read from the vendor strings
  and shown as SUGGESTIONS. They are never written to the sidecar behind your
  back. Look at them; they are usually right, and when they are wrong it is
  because a site typed something unusual into the protocol.
* **format**: `DICOM` or `ECAT`, so you can see at a glance what came from
  where.

Select a row and the Properties panel on the right shows what BIDS will let that
file carry, at the level the standard sets: a red `*` for required, an amber dot
for recommended.

---

## 4. What the conversion already answers

Open the Properties panel for a PET row and look at the green **Already
answered by the conversion** block at the top of each section. That is what was
read out of your files during the scan, with the values that will be written.

On the phantom set a DICOM PET sidecar comes out with about 30 fields filled
before anyone types anything: manufacturer, model, institution, frame timing,
decay correction, attenuation and scatter correction, radionuclide, units,
reconstruction method with its iterations and subsets, and `TimeZero`.

The block is editable, and that is deliberate. If your scanner wrote something
wrong, this is where you correct it. Typing nothing there changes nothing.

**ECAT gets the same treatment.** ECAT7 headers carry roughly sixty fields, and
they are read: scanner model, reconstruction method, per-frame decay and scale
factors, the dose calibration factor, the image units and `TimeZero`. An ECAT
sidecar comes out with 13 to 23 fields depending on what the file records.

One thing to know: if the header records an impossible acquisition time, which
happens on old files, `TimeZero` is deliberately left blank rather than derived
from nonsense. It then appears as a question, because a wrong `TimeZero` makes
every frame time and every blood sample wrong and nothing downstream would catch
it.

---

## 5. Fill the template once for the whole study

This is the part that saves real time. Open **Dataset metadata**.

The dialog is one section per KIND of file, not per file. A section headed

> **every `*_pet.json`** · in `pet/` · 20 files · e.g. `sub-001_trc-FDG_pet.json`

is a statement about all 20 PET runs. Answer the injected dose once and it is
written to every one of them.

For PET you will see roughly 45 questions, of which 10 are required by BIDS.
That sounds like a lot; most of it is one protocol answered once. Work through
the required ones first, they are marked with a red `*`.

Every field has the standard's own description as a tooltip, on the label and on
the box, so you do not have to keep the specification open beside you.

**When something genuinely differs per recording**, leave the dataset answer
blank and set it on the row instead: select the row, and the Properties panel
takes an answer for that recording alone. A per-row answer always beats the
dataset one. If the scan found that your files disagree about a field, the
dialog says `differs per recording` rather than asking you to pick one.

---

## 6. Blood data

Blood is what makes PET quantitative, and BIDS Manager reads PMOD `.bld`
exports.

Select the PET run the blood belongs to, and in the Properties panel find
**BLOOD SAMPLING**. Three series, each linked separately:

| series | what it is |
|---|---|
| Whole blood | activity in whole blood |
| Plasma | activity in plasma |
| Parent fraction | the fraction still unmetabolised |

Next to each, say whether the samples were drawn **manual** or **automatic**.
This is not a formality. Hand-drawn and autosampled series have different time
resolution, so BIDS writes them to separate files, and the choice decides which
file each curve lands in. It is asked here because otherwise the conversion
would have to stop and ask on a console you cannot see.

Link the files, then convert. You get, beside the image:

```
sub-001_trc-FDG_run-1_recording-manual_blood.tsv
sub-001_trc-FDG_run-1_blood.json
```

The names carry the run's own entities, so a subject with two tracers or three
runs gets three distinct sets rather than three files fighting over one name.

Mixed sampling works too. If whole blood came off an autosampler and the rest
was drawn by hand, you get a `recording-automatic` table and a
`recording-manual` table, sharing one sidecar that describes both.

**What the sidecar says** is generated from the table that was actually written:
the columns present, described in the standard's own words, and the
`WholeBloodAvail`, `PlasmaAvail` and `MetaboliteAvail` flags set from what you
actually linked. You are not asked for those three, because they are facts about
what you supplied rather than opinions.

**What you ARE asked**, once you link blood, is a new `every *_blood.json`
section in the dataset dialog: `DispersionCorrected` (required by BIDS),
`WithdrawalRate`, `TubingType`, `TubingLength`, `Haematocrit`, `BloodDensity`
and `DispersionConstant`. If you linked a parent-fraction curve, it also asks
how metabolites were measured, because that requirement only applies when
metabolite data exists.

Only PMOD `.bld` is read. If your lab exports something else, the blood section
will not help you yet.

---

## 7. Convert

Press **Run**.

Conversion and metadata are two steps and the GUI runs both, but it is worth
knowing which does what, because it explains where your answers appear:

* **convert** produces a faithful conversion and repairs only what the converter
  itself got wrong. Your template is not applied here.
* **metadata** applies what you stated: the dataset answers, the per-row
  answers, `dataset_description.json`, `participants.tsv`.

So if a value you typed is not in the file, the metadata step is what puts it
there, and running convert alone gives you a conversion with no opinions in it.

Afterwards, switch to the **Editor** tab, open the dataset, and press
**Validate dataset**. Anything still missing is listed with the fix button
taking you to the field that needs it.

---

## 8. If your doses live in a spreadsheet

Most labs keep injected dose, mass and timing in a spreadsheet rather than
typing them per subject. Point the converter at it:

```
bidsmgr-convert <inventory.tsv> <bids_parent> --pet-spreadsheet doses.xlsx
```

It is loose about column naming and strict about values, so you do not have to
reformat your sheet, but it will not silently accept a dose it cannot parse.

---

## 9. The same thing from the command line

Everything above works headless, which is what you want for a re-run:

```bash
bidsmgr-create  /data/bids/my_pet_study --name "FDG dose escalation"
bidsmgr-scan    /data/raw/pet  /data/scratch/inv.tsv --probe-convert -j 8
# edit inv.tsv, or use the GUI, then:
bidsmgr-convert /data/scratch/inv.tsv /data/bids --raw-root /data/raw/pet -j 8
bidsmgr-metadata /data/bids
bidsmgr-validate /data/bids
```

Two flags matter for PET specifically:

* `--raw-root` is REQUIRED for ECAT if the inventory does not sit beside the raw
  data. ECAT rows store a path relative to the scanned folder, and without this
  the file cannot be found. You get a message telling you so.
* `--pet-spreadsheet` as in section 8.

---

## 10. Things that will bite you, and what they mean

**"Refusing to convert: these recordings would overwrite one another."**
Two rows resolved to the same BIDS name. The message names the files. Nothing
was written. Change an entity on one of them, or exclude it.

**A required field is still missing after conversion.**
Almost always something no scanner records, such as injected mass. Open the
dataset dialog and answer it once for the study.

**A tracer or radionuclide looks wrong in the table.**
Those columns are suggestions read from vendor strings, and they are never
written on their own. Set the real value in the template; the suggestion is
only there to save you typing.

**An ECAT file converted but its sidecar is thin.**
Some ECAT headers genuinely carry little. Check the green block to see what was
read, and fill the rest through the template.

**The CT half of a PET/CT study is missing.**
Deliberate. BIDS has no CT datatype, so it is excluded with a reason rather than
written somewhere invalid.

---

## 11. Relationship to pet2bids

BIDS Manager uses [pypet2bids](https://github.com/openneuropet/PET2BIDS) as a
dependency, for the parts that group solved first and solved well: reading ECAT7
headers, parsing PMOD blood exports, and going back to the DICOM header for
fields dcm2niix leaves out. Those are the reference implementations from the
people who wrote this part of the standard, and reimplementing them to avoid a
dependency would be work spent going backwards.

What BIDS Manager adds is the workflow around them: the inventory table, a
template that asks only what your data has not already answered, per-recording
overrides, validation, and the viewers. It also filters what it takes, so that
only fields BIDS actually declares reach your sidecars, and placeholders never
do.

`tools/compare_pet2bids.py` re-runs the field-by-field comparison over any
dataset, if you want to check that for yourself rather than take it on trust.
