"""What the guide offers, and the steps of each thing it offers.

The guide (switched on in the header, beside Settings) asks the user what
they want to do in the view they are in, and walks them through it one
control at a time. Everything here is plain data and pure functions over the
facts each view reports about itself (``guide_facts``); controls are named by
a target the view resolves (``guide_targets``). No Qt, so every task and every
step can be tested without a window.

- :func:`tasks` lists what a view offers now (a :class:`Task` each, the
  path's usual next one ``recommended``). Home offers creating, opening,
  reopening and practising; the Converter scanning, checking the table,
  metadata per recording and for the dataset, bulk edits, columns,
  converting, the Editor and the settings; the Editor validating, fixing,
  editing a file, renaming, sessions, deleting, anonymising, image quality,
  repairs and undo.
- A task is an ordered list of :class:`TaskStep`. The user moves with Next
  and Previous; a step also moves on by itself when something happens AFTER
  it was shown (:func:`should_advance` compares the facts then and now: a
  folder picked, a scan finished, a row clicked). A fact that was already
  true never skips a step: what the user chose is where the guide starts.
- :func:`shown` is what the card shows for a step: its busy form while the
  app works on that control (a scan, a conversion, a validation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Union

HOME = "welcome"
CONVERTER = "converter"
EDITOR = "editor"

Facts = Mapping[str, object]
Text = Union[str, Callable[[Facts], str]]


@dataclass(frozen=True)
class Step:
    """One step as the card shows it."""

    key: str            # identity: a new key replays the ring's entrance
    target: str         # a name the view (or the header) resolves to a widget
    title: str
    text: str
    number: int = 0     # its place in its task; 0 for a task of one step
    total: int = 0
    busy: bool = False  # the app is working on this control
    done: bool = False  # the task's result: drawn in the success colour
    #: Targets the card must not cover: what its text sends the user to.
    keep_clear: tuple[str, ...] = ()
    #: The task, named above the step.
    kicker: str = ""

    @property
    def counter(self) -> str:
        return f"Step {self.number} of {self.total}" if self.number else ""

    @property
    def heading(self) -> str:
        """The card's first line: the task and the step's place in it."""
        return " · ".join(p for p in (self.kicker, self.counter) if p)


@dataclass(frozen=True)
class TaskStep:
    """One step of a task: where, what, and what moves it on by itself."""

    key: str
    target: str
    title: str
    text: Text
    keep_clear: tuple[str, ...] = ()
    #: While this fact is true the app is working on the control.
    busy_fact: str = ""
    busy_title: str = ""
    busy_text: str = ""
    #: ``(facts when shown, facts now) -> move on``; ``None``: Next only.
    advance: Optional[Callable[[Facts, Facts], bool]] = None
    done: bool = False


@dataclass(frozen=True)
class Task:
    """One thing to do, as the question offers it, and its steps."""

    key: str
    label: str
    detail: str
    steps: tuple[TaskStep, ...]
    recommended: bool = False


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _grew(name: str) -> Callable[[Facts, Facts], bool]:
    """Moves on when the counter ``name`` grew since the step was shown."""
    return lambda then, now: int(now.get(name) or 0) > int(then.get(name) or 0)


def _became(name: str) -> Callable[[Facts, Facts], bool]:
    """Moves on when ``name`` turned true since the step was shown."""
    return lambda then, now: bool(now.get(name)) and not bool(then.get(name))


def _changed(name: str) -> Callable[[Facts, Facts], bool]:
    """Moves on when ``name`` changed to a value since the step was shown."""
    return lambda then, now: bool(now.get(name)) and now.get(name) != then.get(name)


def _select_several(_then: Facts, now: Facts) -> bool:
    return int(now.get("selected_rows") or 0) >= 2


def _keys() -> tuple[str, str]:
    """Shift+click and Ctrl+click (⇧+click, ⌘+click on a Mac)."""
    from ..viz import keynames

    return keynames.mouse(["shift"], "click"), keynames.mouse(["ctrl"], "click")


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


def _home(f: Facts) -> list[Task]:
    opened = _changed("project_root")
    out: list[Task] = []
    if f.get("project"):
        out.append(Task(
            "continue", f"Continue with {f.get('project_name') or 'this dataset'}",
            "Scan, convert and check its data in the Converter.",
            (TaskStep("continue.converter", "converter", "Open the Converter",
                      "The Converter turns raw data into the dataset: choose the "
                      "folder, scan, check the table, convert."),)))
    out.append(Task(
        "create", "Create a new dataset",
        "Choose where it is saved, name it, and create it.",
        (TaskStep("create.browse", "browse", "Choose where to save it",
                  "Pick the folder your new dataset will be saved in. Skip this to "
                  "use the folder shown.", advance=_grew("browse_clicks")),
         TaskStep("create.name", "name", "Name the dataset",
                  "Type a short name for your study, without spaces, for example "
                  "My_Study. It becomes the dataset's folder name.",
                  advance=_became("name")),
         TaskStep("create.create", "create", "Create the dataset",
                  "Press Create dataset. The Converter opens next, ready for your "
                  "raw data.", advance=opened))))
    out.append(Task(
        "open", "Open a dataset you have",
        "A BIDS dataset opens as a project; any other folder can be looked at.",
        (TaskStep("open.folder", "open", "Open its folder",
                  "Pick the dataset's folder. A BIDS dataset opens as a project; any "
                  "other folder can be looked at, or made into one.", advance=opened),)))
    if f.get("recents"):
        out.append(Task(
            "recent", "Reopen a recent project",
            "Pick up where you left off: the table, its edits and its history.",
            (TaskStep("recent.list", "recent", "Choose it",
                      "Double-click a project in Recent projects, or select it and "
                      "press Open.", advance=opened),)))
    out.append(Task(
        "sample", "Practise on a sample dataset",
        "Download a small real dataset to try every step on.",
        (TaskStep("sample.menu", "samples", "Download a sample",
                  "Choose one: it downloads and unpacks. Then create a new dataset "
                  "and scan the sample's folder."),)))
    best = "continue" if f.get("project") else ("recent" if f.get("recents") else "create")
    return _mark(out, best)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def _converter(f: Facts) -> list[Task]:
    if not f.get("project"):
        return [Task("start", "Start on Home", "The Converter works inside a dataset.",
                     (TaskStep("start.home", "home", "Start on Home",
                               "Create a new dataset or open one you have. The "
                               "Converter works inside it."),))]
    rows = bool(f.get("rows"))
    shift, ctrl = _keys()
    scan_steps = (
        TaskStep("scan.folder", "raw_folder", "Choose the raw data folder",
                 "Pick the folder with the recordings: DICOM, EEG, MEG or PET files. "
                 "Its subfolders are included."
                 + (" The table you have stays as an earlier version." if rows else ""),
                 advance=_grew("raw_picks")),
        TaskStep("scan.scan", "scan", "Scan it",
                 "Scan reads the folder and lists every recording it finds, with the "
                 "BIDS name it proposes.", busy_fact="scanning", busy_title="Scanning",
                 busy_text="Reading the raw data. The table fills when the scan is "
                           "done. Stop ends it early.", advance=_grew("tables")),
        TaskStep("scan.table", "table", "Your table",
                 "One row per recording: what will be converted and how. Check it "
                 "next, or convert it.", keep_clear=("properties",), done=True),
    )
    out = [Task("scan", "Scan new data" if rows else "Scan raw data",
                "Another raw data folder, scanned into a new table." if rows
                else "Choose a raw data folder and list its recordings.",
                scan_steps)]
    if rows:
        out += [
            Task("review", "Check and edit the table",
                 "Correct names, leave out what you do not want.",
                 (TaskStep("review.table", "table", "Look through the table",
                           "One row per recording. Untick a row to leave it out; "
                           "double-click a cell to change it. Click a row to see it in "
                           "Properties.", keep_clear=("properties",),
                           advance=_grew("row_clicks")),
                  TaskStep("review.properties", "properties", "Its details in Properties",
                           "The row you clicked: its datatype, suffix and entities, "
                           "checked against BIDS as you type, and the name it will "
                           "get."),
                  TaskStep("review.structure", "filter", "Leave out whole groups",
                           "In this panel, untick a subject, a session or a kind of "
                           "recording to leave all of it out at once."),
                  TaskStep("review.overview", "overview", "See what is unusual",
                           "Overview lists what differs between subjects: a missing "
                           "run, a label spelled two ways, two recordings with one "
                           "name.", done=True))),
            Task("row_meta", "Edit one recording's metadata",
                 "Participant, acquisition and task details, in Properties.",
                 (TaskStep("row_meta.pick", "table", "Choose the recording",
                           "Click its row.", keep_clear=("properties",),
                           advance=_grew("row_clicks")),
                  TaskStep("row_meta.edit", "properties", "Edit it in Properties",
                           "Scroll to its metadata: the participant, and for EEG and "
                           "MEG the line frequency, reference and device; for a "
                           "functional run, its dummy volumes. A value set here wins "
                           "over the dataset's.", done=True))),
        ]
        if f.get("meta_enabled"):
            out.append(Task(
                "dataset_meta", "Edit the dataset's metadata",
                "Device, institution, reference and event labels for every recording.",
                (TaskStep("dataset_meta.open", "dataset_metadata", "Open Dataset metadata",
                          "Values here apply to every recording; a recording's own "
                          "value in Properties wins over them."),)))
        out += [
            Task("bulk", "Edit many rows at once",
                 "One value into one column for every selected row.",
                 (TaskStep("bulk.select", "table", "Select the rows",
                           f"{shift} selects a range; {ctrl} adds or removes one row.",
                           advance=_select_several),
                  TaskStep("bulk.edit", "bulk_edit", "Press Bulk edit",
                           "Choose the column and the value: every selected row takes "
                           "it, as one Undo.", done=True))),
            Task("columns", "Choose the columns shown",
                 "Show or hide columns, each with a note on what it holds.",
                 (TaskStep("columns.open", "columns", "Open Manage columns",
                           "Tick the columns to show. Drag a column's header to move "
                           "it; double-click its edge to fit it."),)),
            Task("convert", "Convert the table",
                 "Write the ticked recordings into the dataset.",
                 (TaskStep("convert.run", "run", "Run the conversion",
                           "Run conversion writes the ticked recordings into the "
                           "dataset with their metadata, then checks it.",
                           busy_fact="converting", busy_title="Converting",
                           busy_text="The log below shows the progress. Stop ends it "
                                     "after the subject in progress.",
                           advance=_grew("conversions")),
                  TaskStep("convert.result", "editor", "Look at the result",
                           "The Editor shows the dataset as written, file by file, and "
                           "validates it.", done=True))),
        ]
    if f.get("subjects") or f.get("converted"):
        out.append(Task("editor", "Look at the dataset",
                        "Open the Editor to check and validate what was converted.",
                        (TaskStep("editor.pill", "editor", "Open the Editor",
                                  "The Editor shows the dataset as written, file by "
                                  "file, and validates it."),)))
    out.append(Task("settings", "Change how scans and conversions work",
                    "Subject labels, dummy volumes, physiology and more.",
                    (TaskStep("settings.open", "settings", "Open Settings",
                              "The Scan and Convert pages hold what a scan proposes and "
                              "what a conversion writes. Changes apply to the next scan "
                              "or conversion."),)))
    if not rows:
        best = "scan"
    elif f.get("converted"):
        best = "editor"
    elif not f.get("row_clicks"):
        best = "review"
    else:
        best = "convert"
    return _mark(out, best)


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


def _findings_text(f: Facts) -> str:
    errors = int(f.get("errors") or 0)
    warnings = int(f.get("warnings") or 0)
    if errors:
        return (f"{_count(errors, 'error')} to fix"
                + (f" and {_count(warnings, 'warning')} to look at" if warnings else "")
                + ". Each finding names its file; Fix opens it where the value is "
                  "edited.")
    if warnings:
        return (f"Nothing BIDS requires is missing. {_count(warnings, 'warning')} "
                "remain, worth a look but not required.")
    return "Nothing BIDS requires is missing, and there are no warnings."


def _editor(f: Facts) -> list[Task]:
    if f.get("view_only"):
        return []                       # a folder to look at, nothing to do
    if not f.get("root"):
        return [Task("open", "Open a dataset", "Look at a dataset's files.",
                     (TaskStep("open.button", "open", "Open a dataset",
                               "Open folder shows a dataset's files. Your own project "
                               "opens here by itself."),))]
    if not f.get("subjects"):
        return [Task("convert_first", "Convert your data first",
                     "This dataset has no subjects yet.",
                     (TaskStep("convert_first.converter", "converter",
                               "Nothing converted yet",
                               "This dataset has no subjects yet. Convert your raw data "
                               "in the Converter first."),))]
    validated = bool(f.get("validated"))
    errors = int(f.get("errors") or 0)
    warnings = int(f.get("warnings") or 0)
    out = [Task(
        "validate", "Validate again" if validated else "Validate the dataset",
        "Check every file against the BIDS standard.",
        (TaskStep("validate.run", "validate", "Validate the dataset",
                  "Validation checks every file against the BIDS standard and lists "
                  "what still needs fixing.", busy_fact="validating",
                  busy_title="Validating",
                  busy_text="Checking every file. The findings appear in the "
                            "Validation panel.", advance=_grew("reports")),
         TaskStep("validate.findings", "findings", "What it found", _findings_text,
                  done=True)))]
    if validated and (errors or warnings):
        out.append(Task(
            "fix", "Fix what validation found",
            ", ".join(p for p in (_count(errors, "error") if errors else "",
                                  _count(warnings, "warning") if warnings else "") if p)
            + ", each a click away from its file.",
            (TaskStep("fix.finding", "findings", "Choose a finding",
                      "Each finding names its file. Fix opens it where the value is "
                      "edited.", advance=_grew("file_opens")),
             TaskStep("fix.edit", "viewer", "Change the value",
                      "Edit it here and Save. Undo takes it back."),
             TaskStep("fix.check", "validate_file", "Check the file again",
                      "Validate file checks just this file; the finding goes when it "
                      "is fixed.", done=True))))
    out += [
        Task("edit", "Look at or edit a file",
             "Its metadata, table or image, from the BIDS tree.",
             (TaskStep("edit.pick", "tree", "Choose a file",
                       "Click a file in the BIDS tree. Its metadata, table or image "
                       "opens in the middle.", advance=_grew("file_opens")),
              TaskStep("edit.change", "viewer", "Look, and change what you need",
                       "A change is saved with Save and taken back with Undo. Validate "
                       "file shows what it still lacks.", done=True))),
        Task("rename", "Rename a subject, session, task or run",
             "Every file, table and link that names it follows.",
             (TaskStep("rename.tree", "tree", "Right-click what to rename",
                       "Right-click a subject's folder or a file and choose Rename. The "
                       "preview shows every file and link that will change."),
              TaskStep("rename.tools", "tools", "Or rename across the dataset",
                       "Tools > Rename entity... renames a value everywhere it is "
                       "used.", done=True))),
        Task("sessions", "Add or remove sessions",
             "Put recordings into a session, or take them out of one.",
             (TaskStep("sessions.tools", "tools", "Open Tools > Sessions",
                       "Choose the recordings and the session label; the preview shows "
                       "where every file goes."),)),
        Task("delete", "Delete recordings",
             "Files, datatypes, sessions or subjects, every reference repaired.",
             (TaskStep("delete.tree", "tree", "Right-click what to delete",
                       "Choose Delete. Tables and links that name it are repaired, and "
                       "one Undo brings it all back."),)),
        Task("anonymise", "Remove identifying information",
             "Names, dates and devices in the metadata; faces in the images.",
             (TaskStep("anonymise.tools", "tools", "Open Tools > Identifiable data",
                       "Anonymise metadata... removes names and IDs and shifts dates; "
                       "Deface... removes faces from anatomical images. Each shows what "
                       "it will change first."),)),
        Task("quality", "Check image quality",
             "A quick check of anatomical and diffusion images.",
             (TaskStep("quality.tools", "tools", "Open Tools > Quality check",
                       "Choose the images; the results open beside them, and are kept "
                       "in derivatives/bidsmgr-qc."),)),
        Task("repair", "Repair sidecars and tables",
             "Missing sidecars, scans tables, names that disagree.",
             (TaskStep("repair.tools", "tools", "Open Tools > Check and repair",
                       "Check coherence... finds what disagrees; Write sidecars from the "
                       "recordings... and Scans and sessions tables... write what is "
                       "missing."),)),
        Task("history", "Undo a change", "Every change made here is one step back.",
             (TaskStep("history.undo", "undo", "Undo",
                       "Undo takes back the last change, Redo puts it back. Right-click "
                       "either for the whole history."),)),
    ]
    best = "validate" if not validated else ("fix" if errors else "edit")
    return _mark(out, best)


# ---------------------------------------------------------------------------
# Asking, showing, moving on
# ---------------------------------------------------------------------------


def _mark(found: list[Task], best: str) -> list[Task]:
    return [Task(t.key, t.label, t.detail, t.steps, t.key == best) for t in found]


def tasks(view: str, facts: Facts) -> list[Task]:
    """What ``view`` offers now, the path's usual next one ``recommended``."""
    if view == HOME:
        return _home(facts)
    if view == CONVERTER:
        return _converter(facts)
    if view == EDITOR:
        return _editor(facts)
    return []


def find_task(view: str, key: str, facts: Facts) -> Optional[Task]:
    """The task ``key`` of ``view`` as the facts now make it, or ``None``
    when it is no longer offered (its table went, say)."""
    return next((t for t in tasks(view, facts) if t.key == key), None)


def shown(task: Task, index: int, facts: Facts) -> Step:
    """What the card shows for ``task``'s step ``index`` with these facts."""
    step = task.steps[index]
    busy = bool(step.busy_fact and facts.get(step.busy_fact))
    text = step.text(facts) if callable(step.text) else step.text
    many = len(task.steps) > 1
    return Step(
        key=step.key + (".busy" if busy else ""),
        target=step.target,
        title=step.busy_title if busy and step.busy_title else step.title,
        text=step.busy_text if busy and step.busy_text else text,
        number=index + 1 if many else 0,
        total=len(task.steps) if many else 0,
        busy=busy,
        done=step.done and not busy,
        keep_clear=step.keep_clear,
        kicker=task.label,
    )


def should_advance(task: Task, index: int, then: Facts, now: Facts) -> bool:
    """Whether step ``index`` moves on by itself: something happened since it
    was shown (``then``). Never while the app is still working on it."""
    step = task.steps[index]
    if step.advance is None or (step.busy_fact and now.get(step.busy_fact)):
        return False
    return bool(step.advance(then, now))


__all__ = ["CONVERTER", "EDITOR", "HOME", "Step", "Task", "TaskStep", "find_task",
           "shown", "should_advance", "tasks"]
