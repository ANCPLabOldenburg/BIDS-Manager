"""Unit tests for ``bidsmgr.editor.html_report``.

The renderer is pure: it takes a ``ValidationReport`` and returns a
self-contained HTML string. Tests assert on the structural shape and
on key escaping / colour-coding choices, not on exact byte sequences.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.editor import (
    FieldLevel,
    FileVerdict,
    Issue,
    Severity,
    SidecarField,
    ValidationReport,
    render_html,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _issue(sev: Severity, rule_id: str, message: str, **kwargs) -> Issue:
    return Issue(severity=sev, rule_id=rule_id, message=message, **kwargs)


def _empty_report(*, severity: Severity = Severity.OK) -> ValidationReport:
    return ValidationReport(
        bids_root=Path("/tmp/study"),
        bidsmgr_version="0.0.1",
        bids_version="1.10.0",
        generated_at="2026-05-09T00:00:00Z",
        severity=severity,
        counts={"ok": 0, "warn": 0, "err": 0},
    )


# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------


class TestSelfContained:
    def test_full_html5_document(self) -> None:
        out = render_html(_empty_report())
        assert out.startswith("<!DOCTYPE html>")
        assert "<html" in out and "</html>" in out
        assert "<head>" in out and "</head>" in out
        assert "<body>" in out and "</body>" in out

    def test_no_external_resources(self) -> None:
        """No <link>/<script>/<img> tags — file must be portable offline."""
        out = render_html(_empty_report())
        assert "<link" not in out.lower()
        assert "<script" not in out.lower()
        # External images would require src; we don't render any.
        assert "<img" not in out.lower()

    def test_inline_style_block_present(self) -> None:
        out = render_html(_empty_report())
        assert "<style>" in out and "</style>" in out
        assert ":root {" in out  # CSS palette block

    def test_uses_meta_charset_utf8(self) -> None:
        out = render_html(_empty_report())
        assert 'meta charset="utf-8"' in out


# ---------------------------------------------------------------------------
# Header / status banner
# ---------------------------------------------------------------------------


class TestHeader:
    def test_renders_dataset_name(self) -> None:
        out = render_html(_empty_report())
        assert "study" in out  # bids_root.name
        assert "BIDS validation" in out

    def test_status_banner_uses_severity_class(self) -> None:
        for sev in [Severity.OK, Severity.WARN, Severity.ERR]:
            out = render_html(_empty_report(severity=sev))
            assert f'class="status {sev.value}"' in out
            assert f'class="label">{sev.value}<' in out

    def test_count_pills_render(self) -> None:
        report = _empty_report()
        report.counts = {"ok": 12, "warn": 3, "err": 1}
        out = render_html(report)
        assert ">12 ok<" in out
        assert ">3 warnings<" in out
        assert ">1 errors<" in out


# ---------------------------------------------------------------------------
# Issue list rendering & color coding
# ---------------------------------------------------------------------------


class TestIssueRendering:
    def test_dataset_issues_render_with_severity_class(self) -> None:
        report = _empty_report()
        report.dataset_issues = [
            _issue(Severity.ERR, "bids.missing_dataset_description",
                   "dataset_description.json is missing"),
        ]
        out = render_html(report)
        # Issue badge picks up the severity class.
        assert '<span class="badge err">err</span>' in out
        assert "missing_dataset_description" in out

    def test_folder_issues_grouped_by_folder(self) -> None:
        report = _empty_report()
        report.folder_issues = {
            "sub-001": [
                _issue(Severity.WARN, "bids.unknown_datatype_dir",
                       "unrecognised directory: 'raw_dicoms'"),
            ],
        }
        out = render_html(report)
        assert "sub-001" in out
        assert "unknown_datatype_dir" in out
        # The folder block carries the worst-severity class.
        assert 'class="folder warn"' in out

    def test_file_issues_render_path_and_typed(self) -> None:
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("sub-001/func/sub-001_task-rest_bold.json"),
                severity=Severity.WARN,
                datatype="func", suffix="bold",
                issues=[
                    _issue(Severity.WARN, "bidsmgr.todo_placeholder",
                           "field 'TaskDescription' contains TODO placeholder",
                           field="TaskDescription"),
                ],
            ),
        ]
        out = render_html(report)
        assert "sub-001/func/sub-001_task-rest_bold.json" in out
        assert "func/bold" in out  # the typed annotation
        assert "TaskDescription" in out

    def test_file_with_no_issues_omitted_from_flagged_section(self) -> None:
        """The "Files with issues" section only shows flagged files.

        OK files now appear in the separate "All files" section
        (added so the HTML carries the same data as the JSON dump).
        """
        report = _empty_report()
        report.files = [
            FileVerdict(path=Path("sub-001/anat/sub-001_T1w.json"),
                        severity=Severity.OK, datatype="anat", suffix="T1w"),
        ]
        out = render_html(report)
        # The "Files with issues" headline still notes that none have issues.
        assert "No per-file issues" in out
        # But the file IS listed in the All files section.
        assert "All files" in out
        assert "sub-001_T1w.json" in out

    def test_color_classes_for_each_severity(self) -> None:
        """Each severity uses the agreed CSS class for color coding."""
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("a.json"),
                severity=Severity.ERR,
                issues=[_issue(Severity.ERR, "rule.x", "boom")],
            ),
            FileVerdict(
                path=Path("b.json"),
                severity=Severity.WARN,
                issues=[_issue(Severity.WARN, "rule.y", "tepid")],
            ),
        ]
        out = render_html(report)
        assert 'class="file err"' in out
        assert 'class="file warn"' in out
        assert '<span class="badge err">err</span>' in out
        assert '<span class="badge warn">warn</span>' in out


# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------


class TestEmptyStates:
    def test_no_dataset_issues_shows_empty_message(self) -> None:
        out = render_html(_empty_report())
        assert "No dataset-level issues" in out

    def test_no_folder_issues_shows_empty_message(self) -> None:
        out = render_html(_empty_report())
        assert "No folder-level issues" in out

    def test_no_file_issues_shows_empty_message(self) -> None:
        out = render_html(_empty_report())
        assert "No per-file issues" in out


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


class TestEscaping:
    def test_escapes_html_in_messages(self) -> None:
        report = _empty_report()
        report.dataset_issues = [
            _issue(Severity.WARN, "rule.x",
                   "<script>alert('xss')</script> & friends"),
        ]
        out = render_html(report)
        # The HTML must NOT contain a real script tag.
        assert "<script>alert" not in out
        # The escaped form is present.
        assert "&lt;script&gt;" in out
        assert "&amp; friends" in out

    def test_escapes_paths_with_special_chars(self) -> None:
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("sub-001/odd<dir>/file.json"),
                severity=Severity.WARN,
                issues=[_issue(Severity.WARN, "rule", "msg")],
            ),
        ]
        out = render_html(report)
        # The dangerous brackets are escaped.
        assert "<dir>" not in out
        assert "&lt;dir&gt;" in out


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestRenderingSize:
    def test_renders_without_crashing_for_realistic_size(self) -> None:
        """100 files × 5 issues each. Just confirm it doesn't blow up."""
        report = _empty_report(severity=Severity.WARN)
        report.counts = {"ok": 100, "warn": 500, "err": 0}
        report.files = [
            FileVerdict(
                path=Path(f"sub-{i:03d}/func/sub-{i:03d}_bold.json"),
                severity=Severity.WARN,
                datatype="func", suffix="bold",
                issues=[
                    _issue(Severity.WARN, f"rule.{j}", f"message {j}",
                           field=f"Field{j}")
                    for j in range(5)
                ],
            )
            for i in range(100)
        ]
        out = render_html(report)
        # Rough size check — should be under a reasonable upper bound.
        assert len(out) < 5_000_000  # 5 MB ceiling for 500 issues
        # 100 files in "Files with issues" + 100 in "All files" = 200.
        assert out.count('<div class="file') == 100
        assert out.count('<details class="file-all"') == 100


# ---------------------------------------------------------------------------
# "All files" + schema-audit table
# ---------------------------------------------------------------------------


class TestAllFilesSection:
    def test_lists_every_file_including_ok_ones(self) -> None:
        """The All files section enumerates each FileVerdict, even
        those that passed (Severity.OK with no issues)."""
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("sub-001/anat/sub-001_T1w.json"),
                severity=Severity.OK,
                datatype="anat", suffix="T1w",
            ),
            FileVerdict(
                path=Path("sub-001/anat/sub-001_T1w.nii.gz"),
                severity=Severity.OK,
                datatype="anat", suffix="T1w",
            ),
        ]
        out = render_html(report)
        assert "All files" in out
        # Two collapsible <details> rows in the All files section.
        assert out.count('<details class="file-all"') == 2
        assert "sub-001_T1w.json" in out
        assert "sub-001_T1w.nii.gz" in out

    def test_section_summary_mentions_total_count(self) -> None:
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path(f"sub-001/anat/file{i}.json"),
                severity=Severity.OK,
            )
            for i in range(5)
        ]
        out = render_html(report)
        assert "5 total" in out

    def test_empty_report_omits_all_files_rows_but_keeps_section(self) -> None:
        out = render_html(_empty_report())
        assert "All files" in out
        # No file rows because there are no files.
        assert '<details class="file-all"' not in out


class TestSchemaAuditTable:
    def test_renders_sidecar_fields_table_for_json_with_audit(self) -> None:
        """JSON files with sidecar_fields get a schema-audit table
        inside their expandable body."""
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("sub-001/anat/sub-001_T1w.json"),
                severity=Severity.WARN,
                datatype="anat", suffix="T1w",
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="MagneticFieldStrength",
                        value=None,
                        present=False,
                        value_kind="missing",
                    ),
                    SidecarField(
                        level=FieldLevel.RECOMMENDED,
                        name="RepetitionTime",
                        value=2.0,
                        present=True,
                        value_kind="number",
                    ),
                ],
            ),
        ]
        out = render_html(report)
        assert '<table class="sidecar-fields"' in out
        # Required-level row carries the lvl-req CSS class.
        assert "lvl-req" in out
        # Missing required field shows the "(missing)" marker.
        assert "(missing)" in out
        # Field names and values appear in the HTML.
        assert "MagneticFieldStrength" in out
        assert "RepetitionTime" in out

    def test_non_json_file_has_no_audit_table(self) -> None:
        """A NIfTI file (no sidecar_fields) has no audit table — the
        body shows a hint instead."""
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("sub-001/anat/sub-001_T1w.nii.gz"),
                severity=Severity.OK,
                datatype="anat", suffix="T1w",
            ),
        ]
        out = render_html(report)
        assert '<table class="sidecar-fields"' not in out
        # Friendly hint instead.
        assert "non-JSON file" in out

    def test_field_description_lands_as_hover_tooltip(self) -> None:
        """Schema field descriptions render as ``title`` attributes on
        the Field cell — hover reveals the docstring without bloating
        the table to a five-column layout."""
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("x.json"),
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="RepetitionTime",
                        value=2.0, present=True,
                        value_kind="number",
                        description="The time, in seconds, between two volumes.",
                    ),
                    SidecarField(
                        level=FieldLevel.OPTIONAL,
                        name="NoDocs",
                        value="x", present=True, value_kind="string",
                        # No description.
                    ),
                ],
            ),
        ]
        out = render_html(report)
        # Cell with a description carries title + has-desc class.
        assert 'class="has-desc"' in out
        assert (
            'title="The time, in seconds, between two volumes."' in out
        )
        # The cell without a description doesn't.
        # (We don't pin a precise byte sequence — just that the
        # description-less field still renders.)
        assert "NoDocs" in out

    def test_levels_sorted_required_first(self) -> None:
        """The schema-audit table puts required fields first regardless
        of disk order."""
        report = _empty_report()
        report.files = [
            FileVerdict(
                path=Path("x.json"),
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.OPTIONAL,
                        name="zzz_opt", value="x", present=True,
                        value_kind="string",
                    ),
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="zzz_req", value="y", present=True,
                        value_kind="string",
                    ),
                ],
            ),
        ]
        out = render_html(report)
        # zzz_req appears before zzz_opt in the document.
        assert out.index("zzz_req") < out.index("zzz_opt")
