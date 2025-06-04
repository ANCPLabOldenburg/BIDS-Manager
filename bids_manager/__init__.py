import sys
import os
import subprocess
import json
import re
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout,
    QTextEdit, QTreeView, QFileSystemModel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QMessageBox, QAction, QSplitter, QDialog, QAbstractItemView)
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QPalette, QColor, QFont
import logging  # debug logging

# Import the scan function directly to ensure TSV generation works
from dicom_inventory import scan_dicoms_long

__version__ = "0.1.0"

# ---- basic logging config ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

class BIDSManager(QMainWindow):
    """
    Main GUI for BIDS Manager.
    Provides two tabs: Convert (DICOM→BIDS pipeline) and Edit (BIDS dataset explorer/editor).
    Supports Windows and Linux.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIDS Manager")
        self.resize(1400, 900)

        # Paths
        self.dicom_dir = ""         # Raw DICOM directory
        self.bids_out_dir = ""      # Output BIDS directory
        self.tsv_path = ""          # Path to subject_summary.tsv
        self.heuristic_path = ""    # Path to auto_heuristic.py

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Tab widget
        self.tabs = QTabWidget()
        # Use larger font for tab labels
        font = QFont()
        font.setPointSize(10)
        self.tabs.setFont(font)
        main_layout.addWidget(self.tabs)

        # Initialize tabs
        self.initConvertTab()
        self.initEditTab()

    def initConvertTab(self):
        """
        Set up Convert tab with 4-section layout and header.
        """
        self.convert_tab = QWidget()
        # Four sections: header (Input/Output), top-left (TSV table), top-right (modalities menu), bottom-left (preview), bottom-right (placeholder)
        main_layout = QVBoxLayout(self.convert_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Header: Input & Output, stacked vertically
        header_group = QGroupBox("Configuration")
        header_layout = QVBoxLayout(header_group)
        header_layout.setSpacing(6)

        # DICOM directory (stacked)
        dicom_label = QLabel("<b>DICOM Dir:</b>")
        self.dicom_dir_edit = QLineEdit()
        self.dicom_dir_edit.setReadOnly(True)
        dicom_browse = QPushButton("Browse…")
        dicom_browse.setFixedWidth(80)
        dicom_browse.clicked.connect(self.selectDicomDir)
        dicom_row = QHBoxLayout()
        dicom_row.addWidget(dicom_label)
        dicom_row.addWidget(self.dicom_dir_edit)
        dicom_row.addWidget(dicom_browse)
        header_layout.addLayout(dicom_row)

        # BIDS output directory (stacked)
        bids_label = QLabel("<b>BIDS Out Dir:</b>")
        self.bids_out_edit = QLineEdit()
        self.bids_out_edit.setReadOnly(True)
        bids_browse = QPushButton("Browse…")
        bids_browse.setFixedWidth(80)
        bids_browse.clicked.connect(self.selectBIDSOutDir)
        bids_row = QHBoxLayout()
        bids_row.addWidget(bids_label)
        bids_row.addWidget(self.bids_out_edit)
        bids_row.addWidget(bids_browse)
        header_layout.addLayout(bids_row)

        # Output folder name (editable)
        name_label = QLabel("<b>Output Name:</b>")
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Enter folder name")
        self.output_name.setFixedWidth(200)
        name_row = QHBoxLayout()
        name_row.addWidget(name_label)
        name_row.addWidget(self.output_name)
        header_layout.addLayout(name_row)

        main_layout.addWidget(header_group, stretch=0)

        # Add a log output area for Convert tab
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        # Remove log group here; will add at end
        # self.log_text = QTextEdit()
        # self.log_text.setReadOnly(True)
        # log_layout.addWidget(self.log_text)
        # main_layout.addWidget(log_group, stretch=0)

        # Create a grid layout for four sections
        grid_widget = QWidget()
        grid_layout = QSplitter(Qt.Vertical)

        # Top half splitter (horizontal)
        top_splitter = QSplitter(Qt.Horizontal)
        # Top-left: TSV Table
        tsv_group = QGroupBox("TSV Viewer")
        tsv_layout = QVBoxLayout(tsv_group)
        # Button to generate TSV
        self.tsv_button = QPushButton("Generate TSV")
        self.tsv_button.setFixedWidth(100)
        self.tsv_button.clicked.connect(self.runInventory)
        tsv_layout.addWidget(self.tsv_button)
        # TSV table
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(6)
        self.mapping_table.setHorizontalHeaderLabels([
            "Include", "Subject", "Session", "Sequence", "Modality", "BIDS Modality"
        ])
        hdr = self.mapping_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.mapping_table.verticalHeader().setVisible(False)
        tsv_layout.addWidget(self.mapping_table)
        top_splitter.addWidget(tsv_group)

        # Top-right: Modalities menu with tabs
        modal_group = QGroupBox("Modalities")
        modal_layout = QVBoxLayout(modal_group)
        # Tabs: full view vs unique
        self.modal_tabs = QTabWidget()
        # Full view tab
        full_tab = QWidget()
        full_layout = QVBoxLayout(full_tab)
        self.full_tree = QTreeWidget()
        self.full_tree.setHeaderLabels(["Modality/NON-BIDS","Sequences"])
        full_layout.addWidget(self.full_tree)
        self.modal_tabs.addTab(full_tab, "Full View")
        # Unique tab
        unique_tab = QWidget()
        unique_layout = QVBoxLayout(unique_tab)
        self.unique_tree = QTreeWidget()
        self.unique_tree.setHeaderLabels(["Sequence"])
        unique_layout.addWidget(self.unique_tree)
        self.modal_tabs.addTab(unique_tab, "Unique")
        modal_layout.addWidget(self.modal_tabs)
        top_splitter.addWidget(modal_group)

        grid_layout.addWidget(top_splitter)

        # Bottom half splitter (horizontal)
        bottom_splitter = QSplitter(Qt.Horizontal)
        # Bottom-left: Preview tree
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["BIDS Structure"])
        preview_layout.addWidget(self.preview_tree)
        bottom_splitter.addWidget(preview_group)

        # Bottom-right: Placeholder for future controls
        placeholder_group = QGroupBox("Settings")
        placeholder_layout = QVBoxLayout(placeholder_group)
        placeholder_label = QLabel("Additional controls can go here.")
        placeholder_layout.addWidget(placeholder_label)
        bottom_splitter.addWidget(placeholder_group)

        grid_layout.addWidget(bottom_splitter)
        main_layout.addWidget(grid_layout, stretch=1)

        # Action buttons row
        actions_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.setFixedWidth(100)
        self.preview_button.clicked.connect(self.updateModalTree)
        self.run_button = QPushButton("Run")
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group, stretch=0)

        self.run_button.setFixedWidth(100)
        self.run_button.clicked.connect(self.runFullConversion)
        actions_layout.addStretch()
        actions_layout.addWidget(self.preview_button)
        actions_layout.addWidget(self.run_button)
        main_layout.addLayout(actions_layout)

        self.tabs.addTab(self.convert_tab, "Convert")

    def updateModalTree(self):
        logging.info("updateModalTree → Building preview tree …")
        """Populate preview tree based on checked sequences."""
        self.preview_tree.clear()
        for i in range(self.mapping_table.rowCount()):
            include = (self.mapping_table.item(i, 0).checkState() == Qt.Checked)
            if include:
                subj = self.mapping_table.item(i, 1).text()
                ses = self.mapping_table.item(i, 2).text()
                seq = self.mapping_table.item(i, 3).text()
                modb = self.mapping_table.item(i, 5).text()
                item = QTreeWidgetItem([f"{subj}/{ses}/{modb}/{subj}_{ses}_{seq}.nii.gz"])
                self.preview_tree.addTopLevelItem(item)
        self.preview_tree.expandAll()

    # (Rest of code remains unchanged)

    def initEditTab(self):
        """
        Set up Edit tab to embed the full functionality of bids_editor_ancpbids.
        """
        self.edit_tab = QWidget()
        edit_layout = QVBoxLayout(self.edit_tab)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)

        # Menu bar for Edit features
        self.menuBar().clear()
        file_menu = self.menuBar().addMenu("File")
        open_act = QAction("Open BIDS…", self)
        open_act.triggered.connect(self.openBIDSForEdit)
        file_menu.addAction(open_act)
        tools_menu = self.menuBar().addMenu("Tools")
        rename_act = QAction("Batch Rename…", self)
        rename_act.triggered.connect(self.launchBatchRename)
        tools_menu.addAction(rename_act)

        # Splitter between left (tree & stats) and right (metadata)
        splitter = QSplitter()
        splitter.setHandleWidth(4)

        # Left panel: BIDSplorer and BIDStatistics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel('<b>BIDSplorer</b>'))
        self.model = QFileSystemModel()
        self.model.setRootPath("")
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(3, True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        left_layout.addWidget(self.tree)
        self.tree.clicked.connect(self.onTreeClicked)

        left_layout.addWidget(QLabel('<b>BIDStatistics</b>'))
        self.stats = QTreeWidget()
        self.stats.setHeaderLabels(["Metric", "Value"])
        self.stats.setAlternatingRowColors(True)
        s_hdr = self.stats.header()
        s_hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        s_hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        left_layout.addWidget(self.stats)

        splitter.addWidget(left_panel)

        # Right panel: MetadataViewer (reused from original)
        self.viewer = MetadataViewer()
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 2)

        edit_layout.addWidget(splitter)
        self.tabs.addTab(self.edit_tab, "Edit")

    def selectDicomDir(self):
        """Select the raw DICOM input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if directory:
            self.dicom_dir = directory
            self.dicom_dir_edit.setText(directory)

    def selectBIDSOutDir(self):
        """Select (or create) the BIDS output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select/Create BIDS Output Directory")
        if directory:
            self.bids_out_dir = directory
            self.bids_out_edit.setText(directory)

    def runInventory(self):
        logging.info("runInventory → Generating TSV …")
        """
        Scan DICOMs and generate subject_summary.tsv in a dedicated output folder.
        The final output path is <BIDS Out Dir>/<Output Name>/subject_summary.tsv.
        """
        if not self.dicom_dir or not os.path.isdir(self.dicom_dir):
            QMessageBox.warning(self, "Invalid DICOM Directory", "Please select a valid DICOM input directory.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output Directory", "Please select a BIDS output directory.")
            return

        # Resolve final output directory (may include sub‑folder name)
        out_name = self.output_name.text().strip()
        self.final_out_dir = os.path.join(self.bids_out_dir, out_name) if out_name else self.bids_out_dir
        os.makedirs(self.final_out_dir, exist_ok=True)

        self.tsv_path = os.path.join(self.final_out_dir, "subject_summary.tsv")

        # Generate TSV via dicom_inventory
        try:
            scan_dicoms_long(self.dicom_dir, self.tsv_path)
            self.log_text.append(f"TSV generated at {self.tsv_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"dicom_inventory failed: {e}")
            return

        self.loadMappingTable()

    def loadMappingTable(self):
        logging.info("loadMappingTable → Loading TSV into table …")
        """
        Load the generated TSV into the mapping_table for user editing.
        Columns: include, subject, session, sequence, modality, modality_bids
        """
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            return
        df = pd.read_csv(self.tsv_path, sep="\t")

        # Populate table rows
        self.mapping_table.setRowCount(0)
        for _, row in df.iterrows():
            r = self.mapping_table.rowCount()
            self.mapping_table.insertRow(r)
            # Include: checkbox
            include_item = QTableWidgetItem()
            include_item.setFlags(include_item.flags() | Qt.ItemIsUserCheckable)
            include_item.setCheckState(Qt.Checked if row.get('include', 1) == 1 else Qt.Unchecked)
            self.mapping_table.setItem(r, 0, include_item)
            # Subject (non-editable)
            subj_item = QTableWidgetItem(str(row.get('BIDS_name', '')))
            subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 1, subj_item)
            # Session (non-editable)
            ses_item = QTableWidgetItem(str(row.get('session', '')))
            ses_item.setFlags(ses_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 2, ses_item)
            # Sequence (editable)
            seq_item = QTableWidgetItem(str(row.get('sequence', '')))
            seq_item.setFlags(seq_item.flags() | Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 3, seq_item)
            # Modality (non-editable)
            mod_item = QTableWidgetItem(str(row.get('modality', '')))
            mod_item.setFlags(mod_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 4, mod_item)
            # BIDS Modality (editable)
            modb_item = QTableWidgetItem(str(row.get('modality_bids', '')))
            modb_item.setFlags(modb_item.flags() | Qt.ItemIsEditable)
            self.mapping_table.setItem(r, 5, modb_item)
        self.log_text.append("Loaded TSV into mapping table.")
        # Populate modalities trees immediately
        self.full_tree.clear()
        self.unique_tree.clear()
        mod_map = {}
        for i in range(self.mapping_table.rowCount()):
            seq = self.mapping_table.item(i, 3).text()
            mod = self.mapping_table.item(i, 4).text()
            mod_map.setdefault(mod, set()).add(seq)
        for mod, seqs in mod_map.items():
            mod_item = QTreeWidgetItem([mod])
            for seq in seqs:
                seq_item = QTreeWidgetItem(["", seq])
                mod_item.addChild(seq_item)
            self.full_tree.addTopLevelItem(mod_item)
        unique_seqs = set()
        for seqs in mod_map.values():
            unique_seqs |= seqs
        for seq in sorted(unique_seqs):
            self.unique_tree.addTopLevelItem(QTreeWidgetItem([seq]))
        self.full_tree.expandAll()
        self.unique_tree.expandAll()

    def previewBIDS(self):
        """
        Show a simple preview text of how selected sequences will map to BIDS structure.
        """
        rows = self.mapping_table.rowCount()
        lines = []
        for i in range(rows):
            include = (self.mapping_table.item(i, 0).checkState() == Qt.Checked)
            subj = self.mapping_table.item(i, 1).text()
            ses = self.mapping_table.item(i, 2).text()
            seq = self.mapping_table.item(i, 3).text()
            modb = self.mapping_table.item(i, 5).text()
            if include:
                # Construct a sample BIDS path
                path = f"{subj}/{ses}/{modb}/{subj}_{ses}_{seq}.nii.gz"
                lines.append(path)
        self.preview_text.setPlainText("\n".join(lines))
        self.log_text.append("Preview generated.")

    def runFullConversion(self):
        logging.info("runFullConversion → Starting full pipeline …")
        """
        Execute full pipeline:
        1) Save updated TSV
        2) build_heuristic_from_tsv.py → auto_heuristic.py
        3) run_heudiconv_from_heuristic.py to convert DICOMs
        4) post_conv_renamer.py on BIDS output
        """
        if not self.tsv_path or not os.path.isfile(self.tsv_path):
            QMessageBox.warning(self, "No TSV", "Please generate the TSV first.")
            return
        if not self.bids_out_dir:
            QMessageBox.warning(self, "No BIDS Output", "Please select a BIDS output directory.")
            return

        # 1) Save updated TSV from table
        try:
            df_updated = []
            for i in range(self.mapping_table.rowCount()):
                include = 1 if self.mapping_table.item(i, 0).checkState() == Qt.Checked else 0
                subj = self.mapping_table.item(i, 1).text()
                ses = self.mapping_table.item(i, 2).text()
                seq = self.mapping_table.item(i, 3).text()
                mod = self.mapping_table.item(i, 4).text()
                modb = self.mapping_table.item(i, 5).text()
                # Build a minimal row; full dicom_inventory already stored other columns
                df_updated.append({
                    'BIDS_name': subj,
                    'session': ses,
                    'include': include,
                    'sequence': seq,
                    'modality': mod,
                    'modality_bids': modb
                })
            # Write back: read original TSV to preserve other columns
            df_orig = pd.read_csv(self.tsv_path, sep="\t")
            # Update only include, sequence, modality_bids columns in df_orig
            for idx, row in enumerate(df_orig.itertuples()):
                df_orig.at[idx, 'include'] = df_updated[idx]['include']
                df_orig.at[idx, 'sequence'] = df_updated[idx]['sequence']
                df_orig.at[idx, 'modality_bids'] = df_updated[idx]['modality_bids']
            df_orig.to_csv(self.tsv_path, sep="\t", index=False)
            self.log_text.append("Saved updated TSV.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save TSV: {e}")
            return

        # Paths for helper scripts (located alongside this package)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        build_script = os.path.join(root_dir, "build_heuristic_from_tsv.py")
        run_script = os.path.join(root_dir, "run_heudiconv_from_heuristic.py")
        rename_script = os.path.join(root_dir, "post_conv_renamer.py")

        # 2) Build heuristic
        self.heuristic_path = os.path.join(self.bids_out_dir, "auto_heuristic.py")
        try:
            proc = subprocess.run([
                sys.executable, build_script,
                self.tsv_path, self.heuristic_path
            ], check=True, capture_output=True, text=True)
            self.log_text.append(proc.stdout)
            self.log_text.append(f"Generated heuristic at {self.heuristic_path}")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"build_heuristic failed:\n{e.stderr}")
            return

        # 3) Run HeuDiConv conversion
        try:
            proc = subprocess.run([
                sys.executable, run_script,
                self.dicom_dir, self.heuristic_path, self.bids_out_dir
            ], check=True, capture_output=True, text=True)
            self.log_text.append(proc.stdout)
            self.log_text.append(f"HeuDiConv conversion complete.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"run_heudiconv failed:\n{e.stderr}")
            return

        # 4) Post-conversion fieldmap renaming
        try:
            proc = subprocess.run([
                sys.executable, rename_script, self.bids_out_dir
            ], check=True, capture_output=True, text=True)
            self.log_text.append(proc.stdout)
            self.log_text.append("Fieldmap renaming applied.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"post_conv_renamer failed:\n{e.stderr}")
            return

        self.log_text.append("Conversion pipeline finished successfully.")

    # ----- Edit tab methods (full bids_editor_ancpbids features) -----
    def openBIDSForEdit(self):
        """Prompt user to select a BIDS dataset root for editing."""
        p = QFileDialog.getExistingDirectory(self, "Select BIDS dataset")
        if p:
            self.bids_root = Path(p)
            self.model.setRootPath(p)
            self.tree.setRootIndex(self.model.index(p))
            self.viewer.clear()
            self.updateStats()
            self.setWindowTitle(f"BIDS Manager – {p}")

    def onTreeClicked(self, idx: QModelIndex):
        """When a file is clicked in the tree, load metadata if JSON/TSV."""
        p = Path(self.model.filePath(idx))
        self.selected = p
        if p.suffix.lower() in ['.json', '.tsv']:
            self.viewer.load_file(p)

    def updateStats(self):
        """Compute and display BIDS stats: total subjects, files, modalities."""
        root = self.bids_root
        self.stats.clear()
        subs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total subjects", str(len(subs))]))
        files = list(root.rglob('*.*'))
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total files", str(len(files))]))
        for sub in subs:
            si = QTreeWidgetItem([sub.name, ""])
            sessions = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith('ses-')]
            if len(sessions) > 1:
                for ses in sessions:
                    s2 = QTreeWidgetItem([ses.name, ""])
                    mods = set(p.parent.name for p in ses.rglob('*.nii*'))
                    s2.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                    for m in mods:
                        imgs = len(list(ses.rglob(f'{m}/*.*')))
                        meta = len(list(ses.rglob(f'{m}/*.json'))) + len(list(ses.rglob(f'{m}/*.tsv')))
                        s2.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
                    si.addChild(s2)
            else:
                mods = set(p.parent.name for p in sub.rglob('*.nii*'))
                si.addChild(QTreeWidgetItem(["Sessions", "1"]))
                si.addChild(QTreeWidgetItem(["Modalities", str(len(mods))]))
                for m in mods:
                    imgs = len(list(sub.rglob(f'{m}/*.*')))
                    meta = len(list(sub.rglob(f'{m}/*.json'))) + len(list(sub.rglob(f'{m}/*.tsv')))
                    si.addChild(QTreeWidgetItem([m, f"imgs:{imgs}, meta:{meta}"]))
            self.stats.addTopLevelItem(si)
        self.stats.expandAll()

    def launchBatchRename(self):
        """Open the Batch Rename dialog from bids_editor_ancpbids."""
        dlg = RemapDialog(self, getattr(self, 'selected', self.bids_root))
        dlg.exec_()


class RemapDialog(QDialog):
    """
    Batch Rename Conditions dialog (from bids_editor_ancpbids).
    Allows regex-based renaming across the dataset.
    """
    def __init__(self, parent, default_scope: Path):
        super().__init__(parent)
        self.setWindowTitle("Batch Remap Conditions")
        self.resize(1000, 600)
        self.bids_root = default_scope
        layout = QVBoxLayout(self)

        # Tabs for multiple conditions
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.add_condition()  # initial condition

        # Button to add more conditions
        btn_add_cond = QPushButton("Add Condition")
        btn_add_cond.clicked.connect(self.add_condition)
        layout.addWidget(btn_add_cond)

        # Scope selector (simple text for demo)
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))
        self.scope_combo = QLineEdit()
        self.scope_combo.setText("Entire dataset")
        scope_layout.addWidget(self.scope_combo)
        layout.addLayout(scope_layout)

        # Preview and Apply buttons
        btn_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply)
        btn_layout.addWidget(self.preview_button)
        btn_layout.addWidget(self.apply_button)
        layout.addLayout(btn_layout)

        # Preview tree
        self.preview_tree = QTreeWidget()
        hdr = self.preview_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.preview_tree)

    def add_condition(self):
        """Add a new tab with a table for regex pattern/replacement pairs."""
        tab = QWidget()
        fl = QVBoxLayout(tab)
        rules_tbl = QTableWidget(0, 2)
        rules_tbl.setHorizontalHeaderLabels(["Pattern", "Replacement"])
        hdr = rules_tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        rule_btns = QHBoxLayout()
        btn_addr = QPushButton("Add Rule")
        btn_addr.clicked.connect(lambda: rules_tbl.insertRow(rules_tbl.rowCount()))
        btn_delr = QPushButton("Delete Rule")
        btn_delr.clicked.connect(lambda: rules_tbl.removeRow(rules_tbl.currentRow()))
        rule_btns.addWidget(btn_addr)
        rule_btns.addWidget(btn_delr)
        rule_btns.addStretch()
        fl.addLayout(rule_btns)
        fl.addWidget(rules_tbl)
        index = self.tabs.count() + 1
        self.tabs.addTab(tab, f"Condition {index}")

    def get_scope_paths(self):
        """Retrieve file paths under the selected scope. TODO: filter by actual scope selection."""
        all_files = [p for p in self.bids_root.rglob('*') if p.is_file()]
        return all_files

    def preview(self):
        """Show potential renaming preview in a tree widget."""
        self.preview_tree.clear()
        paths = self.get_scope_paths()
        for path in sorted(paths):
            name = path.name
            new_name = name
            for i in range(self.tabs.count()):
                tbl = self.tabs.widget(i).findChild(QTableWidget)
                for r in range(tbl.rowCount()):
                    pat_item = tbl.item(r, 0)
                    rep_item = tbl.item(r, 1)
                    if pat_item and pat_item.text():
                        new_name = re.sub(pat_item.text(), rep_item.text() if rep_item else "", new_name)
            if new_name != name:
                item = QTreeWidgetItem([path.relative_to(self.bids_root).as_posix(), new_name])
                self.preview_tree.addTopLevelItem(item)
        self.preview_tree.expandAll()

    def apply(self):
        """Apply renaming to files as shown in preview."""
        for i in range(self.preview_tree.topLevelItemCount()):
            it = self.preview_tree.topLevelItem(i)
            orig = self.bids_root / it.text(0)
            new = orig.with_name(it.text(1))
            orig.rename(new)
        QMessageBox.information(self, "Batch Remap", "Rename applied.")
        self.accept()

class MetadataViewer(QWidget):
    """
    Metadata viewer/editor for JSON and TSV sidecars (from bids_editor_ancpbids).
    """
    def __init__(self):
        super().__init__()
        vlay = QVBoxLayout(self)
        self.welcome = QLabel(
            "<h3>Metadata BIDSualizer</h3><br>Load data via File → Open or select a file to begin editing."
        )
        self.welcome.setAlignment(Qt.AlignCenter)
        vlay.addWidget(self.welcome)
        self.toolbar = QHBoxLayout()
        vlay.addLayout(self.toolbar)
        self.viewer = None
        self.current_path = None

    def clear(self):
        """Clear the toolbar and viewer when switching files."""
        while self.toolbar.count():
            w = self.toolbar.takeAt(0).widget()
            if w:
                w.deleteLater()
        if self.viewer:
            self.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        self.welcome.show()

    def load_file(self, path: Path):
        """Load JSON or TSV file into an editable viewer."""
        self.current_path = path
        self.clear()
        self.welcome.hide()
        ext = path.suffix.lower()
        if ext == '.json':
            self._setup_json_toolbar()
            self.viewer = self._json_view(path)
        elif ext == '.tsv':
            self._setup_tsv_toolbar()
            self.viewer = self._tsv_view(path)
        self.layout().addWidget(self.viewer)

    def _setup_json_toolbar(self):
        """Add buttons for JSON editing: Add Field, Delete Field, Save."""
        for txt, fn in [("Add Field", self._add_field), ("Delete Field", self._del_field), ("Save", self._save)]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _setup_tsv_toolbar(self):
        """Add buttons for TSV editing: Add Row, Del Row, Add Col, Del Col, Save."""
        for txt, fn in [
            ("Add Row", self._add_row), ("Del Row", self._del_row),
            ("Add Col", self._add_col), ("Del Col", self._del_col), ("Save", self._save)
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _add_field(self):
        """Insert a new key/value pair into JSON tree."""
        tree = self.viewer
        sel = tree.currentItem() or tree.invisibleRootItem()
        item = QTreeWidgetItem(["newKey", "newValue"])
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        sel.addChild(item)
        tree.editItem(item, 0)

    def _del_field(self):
        """Delete selected field from JSON tree."""
        tree = self.viewer
        it = tree.currentItem()
        if it:
            parent = it.parent() or tree.invisibleRootItem()
            parent.removeChild(it)

    def _add_row(self):
        """Insert a new row into TSV table."""
        tbl = self.viewer
        tbl.insertRow(tbl.rowCount())

    def _del_row(self):
        """Delete selected row from TSV table."""
        tbl = self.viewer
        r = tbl.currentRow()
        if r >= 0:
            tbl.removeRow(r)

    def _add_col(self):
        """Insert a new column into TSV table."""
        tbl = self.viewer
        c = tbl.columnCount()
        tbl.insertColumn(c)
        tbl.setHorizontalHeaderItem(c, QTableWidgetItem(f"col{c+1}"))

    def _del_col(self):
        """Delete selected column from TSV table."""
        tbl = self.viewer
        c = tbl.currentColumn()
        if c >= 0:
            tbl.removeColumn(c)

    def _json_view(self, path: Path) -> QTreeWidget:
        """Create a tree widget to show and edit JSON data."""
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key", "Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        data = json.loads(path.read_text(encoding='utf-8'))
        self._populate_json(tree.invisibleRootItem(), data)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tree

    def _populate_json(self, parent, data):
        """Recursively populate JSON dictionary into the tree widget."""
        if isinstance(data, dict):
            for k, v in data.items():
                it = QTreeWidgetItem([str(k), '' if isinstance(v, (dict, list)) else str(v)])
                it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                it = QTreeWidgetItem([str(i), '' if isinstance(v, (dict, list)) else str(v)])
                it.setFlags(it.flags() | Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v, (dict, list)):
                    self._populate_json(it, v)

    def _tsv_view(self, path: Path) -> QTableWidget:
        """Create a table widget to show and edit TSV data."""
        df = pd.read_csv(path, sep="\t")
        tbl = QTableWidget(df.shape[0], df.shape[1])
        tbl.setAlternatingRowColors(True)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for j, col in enumerate(df.columns):
            tbl.setHorizontalHeaderItem(j, QTableWidgetItem(col))
            for i, val in enumerate(df[col].astype(str)):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                tbl.setItem(i, j, item)
        for j in range(1, tbl.columnCount()):
            hdr.setSectionResizeMode(j, QHeaderView.Interactive)
        tbl.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        return tbl

    def _save(self):
        """Save edits made to JSON or TSV sidecar back to disk."""
        path = self.current_path
        if path.suffix.lower() == '.json':
            data = self._tree_to_obj(self.viewer.invisibleRootItem())
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        else:
            tbl = self.viewer
            hdrs = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
            rows = [{hdrs[c]: tbl.item(r, c).text() for c in range(tbl.columnCount())} for r in range(tbl.rowCount())]
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        QMessageBox.information(self, "Saved", f"Saved {path}")

    def _tree_to_obj(self, root):
        """Convert tree representation back to nested JSON object."""
        obj = {} if root.childCount() else None
        for i in range(root.childCount()):
            ch = root.child(i)
            k, val = ch.text(0), ch.text(1)
            child = self._tree_to_obj(ch)
            obj[k] = child if child is not None else val
        return obj

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(240, 240, 240))
    pal.setColor(QPalette.WindowText, Qt.black)
    app.setPalette(pal)
    win = BIDSManager()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

