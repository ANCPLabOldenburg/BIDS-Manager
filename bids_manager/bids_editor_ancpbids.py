#!/usr/bin/env python3
"""
bids_editor_gui.py — Professional BIDS Explorer & Metadata Editor v4.1
---------------------------------------------------------------------------
Features:
  • File → Open BIDS root.
  • Tools → Batch Rename… with regex replacement.
  • Left pane: editable file tree (Name/Type columns, inline rename).
  • Stats below tree: alternating rows, interactive column widths.
  • Right pane: metadata editor:
    - JSON: Add/Delete Field, Save; editable tree with nested support, alternating rows, interactive columns.
    - TSV : Add/Delete Row/Column, Save; editable table, alternating rows, interactive columns.
"""
import sys, json, re, pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileSystemModel, QTreeView,
    QSplitter, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
    QTreeWidgetItem, QTableWidget, QTableWidgetItem, QPushButton,
    QAction, QFileDialog, QMessageBox, QLabel, QAbstractItemView,
    QHeaderView, QDialog, QFormLayout, QLineEdit, QComboBox, QTabWidget
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QModelIndex

class RemapDialog(QDialog):
    def __init__(self, parent, default_scope: Path):
        super().__init__(parent)
        self.setWindowTitle("Batch Remap Conditions")
        # default dialog size
        self.resize(1000, 600)
        self.bids_root = default_scope
        layout = QVBoxLayout(self)
        # Conditions tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.add_condition()  # initial condition
        # Button to add more conditions
        btn_add_cond = QPushButton("Add Condition")
        btn_add_cond.clicked.connect(self.add_condition)
        layout.addWidget(btn_add_cond)
        # Scope selector
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Entire dataset", "Selected subjects", "Selected sessions", "Selected modalities"])
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
        # equal 50/50 column widths
        hdr = self.preview_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.preview_tree)

    def add_condition(self):
        tab = QWidget()
        fl = QVBoxLayout(tab)
        # Rules table
        rules_tbl = QTableWidget(0,2)
        rules_tbl.setHorizontalHeaderLabels(["Pattern","Replacement"])
        hdr = rules_tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        # Buttons to add/remove rules
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

    def preview(self):
        self.preview_tree.clear()
        paths = self.get_scope_paths()
        for path in sorted(paths):
            name = path.name
            new_name = name
            for i in range(self.tabs.count()):
                tbl = self.tabs.widget(i).findChild(QTableWidget)
                for r in range(tbl.rowCount()):
                    pat_item = tbl.item(r,0)
                    rep_item = tbl.item(r,1)
                    if pat_item and pat_item.text():
                        new_name = re.sub(pat_item.text(), rep_item.text() if rep_item else "", new_name)
            if new_name != name:
                item = QTreeWidgetItem([path.relative_to(self.bids_root).as_posix(), new_name])
                self.preview_tree.addTopLevelItem(item)
        self.preview_tree.expandAll()

    def apply(self):
        for i in range(self.preview_tree.topLevelItemCount()):
            it = self.preview_tree.topLevelItem(i)
            orig = self.bids_root / it.text(0)
            new = orig.with_name(it.text(1))
            orig.rename(new)
        QMessageBox.information(self, "Batch Remap", "Rename applied.")
        self.accept()

    def get_scope_paths(self):
        idx = self.scope_combo.currentIndex()
        all_files = [p for p in self.bids_root.rglob('*') if p.is_file()]
        # TODO: filter by selected subjects/sessions/modalities
        return all_files

class MetadataViewer(QWidget):
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
        while self.toolbar.count():
            w = self.toolbar.takeAt(0).widget()
            if w: w.deleteLater()
        if self.viewer:
            self.layout().removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        self.welcome.show()

    def load_file(self, path: Path):
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
        for txt, fn in [
            ("Add Field", self._add_field),
            ("Delete Field", self._del_field),
            ("Save", self._save)
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _setup_tsv_toolbar(self):
        for txt, fn in [
            ("Add Row", self._add_row),
            ("Del Row", self._del_row),
            ("Add Col", self._add_col),
            ("Del Col", self._del_col),
            ("Save", self._save)
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            self.toolbar.addWidget(btn)
        self.toolbar.addStretch()

    def _add_field(self):
        tree = self.viewer
        sel = tree.currentItem() or tree.invisibleRootItem()
        item = QTreeWidgetItem(["newKey","newValue"])
        item.setFlags(item.flags()|Qt.ItemIsEditable)
        sel.addChild(item)
        tree.editItem(item,0)

    def _del_field(self):
        tree = self.viewer
        it = tree.currentItem()
        if it:
            parent = it.parent() or tree.invisibleRootItem()
            parent.removeChild(it)

    def _add_row(self):
        tbl = self.viewer
        tbl.insertRow(tbl.rowCount())

    def _del_row(self):
        tbl = self.viewer
        r = tbl.currentRow()
        if r>=0: tbl.removeRow(r)

    def _add_col(self):
        tbl = self.viewer
        c = tbl.columnCount()
        tbl.insertColumn(c)
        tbl.setHorizontalHeaderItem(c,QTableWidgetItem(f"col{c+1}"))

    def _del_col(self):
        tbl = self.viewer
        c = tbl.currentColumn()
        if c>=0: tbl.removeColumn(c)

    def _json_view(self, path: Path) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key","Value"])
        tree.setAlternatingRowColors(True)
        hdr = tree.header()
        hdr.setSectionResizeMode(0,QHeaderView.Interactive)
        hdr.setSectionResizeMode(1,QHeaderView.Interactive)
        data = json.loads(path.read_text(encoding='utf-8'))
        self._populate(tree.invisibleRootItem(), data)
        tree.expandAll()
        tree.setEditTriggers(QAbstractItemView.DoubleClicked|QAbstractItemView.EditKeyPressed)
        return tree

    def _populate(self, parent, data):
        if isinstance(data, dict):
            for k,v in data.items():
                it = QTreeWidgetItem([str(k),'' if isinstance(v,(dict,list)) else str(v)])
                it.setFlags(it.flags()|Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v,(dict,list)): self._populate(it,v)
        elif isinstance(data,list):
            for i,v in enumerate(data):
                it = QTreeWidgetItem([str(i),'' if isinstance(v,(dict,list)) else str(v)])
                it.setFlags(it.flags()|Qt.ItemIsEditable)
                parent.addChild(it)
                if isinstance(v,(dict,list)): self._populate(it,v)

    def _tsv_view(self, path: Path) -> QTableWidget:
        df = pd.read_csv(path,sep="\t")
        tbl = QTableWidget(df.shape[0],df.shape[1])
        tbl.setAlternatingRowColors(True)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0,QHeaderView.Interactive)
        for j,col in enumerate(df.columns):
            tbl.setHorizontalHeaderItem(j,QTableWidgetItem(col))
            for i,val in enumerate(df[col].astype(str)):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags()|Qt.ItemIsEditable)
                tbl.setItem(i,j,item)
        for j in range(1,tbl.columnCount()): hdr.setSectionResizeMode(j,QHeaderView.Interactive)
        tbl.setEditTriggers(QAbstractItemView.DoubleClicked|QAbstractItemView.EditKeyPressed)
        return tbl

    def _save(self):
        path = self.current_path
        if path.suffix.lower()=='.json':
            data = self._tree_to_obj(self.viewer.invisibleRootItem())
            path.write_text(json.dumps(data,indent=2),encoding='utf-8')
        else:
            tbl = self.viewer
            hdrs = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
            rows = [{hdrs[c]:tbl.item(r,c).text() for c in range(tbl.columnCount())} for r in range(tbl.rowCount())]
            pd.DataFrame(rows).to_csv(path,sep="\t",index=False)
        QMessageBox.information(self,"Saved",f"Saved {path}")

    def _tree_to_obj(self, root):
        obj = {} if root.childCount() else None
        for i in range(root.childCount()):
            ch = root.child(i)
            k,val = ch.text(0), ch.text(1)
            child = self._tree_to_obj(ch)
            obj[k] = child if child is not None else val
        return obj

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIDS Editor GUI v4.1")
        open_act = QAction("Open BIDS…",self) ; open_act.triggered.connect(self.open_dir)
        rename_act = QAction("Batch Rename…",self) ; rename_act.triggered.connect(self.batch_rename)
        file_menu = self.menuBar().addMenu("File") ; file_menu.addAction(open_act)
        tools_menu = self.menuBar().addMenu("Tools") ; tools_menu.addAction(rename_act)
        splitter = QSplitter()
        left = QWidget(); l = QVBoxLayout(left)
        # Section: BIDSplorer
        l.addWidget(QLabel('<b>BIDSplorer</b>'))
        self.model = QFileSystemModel(); self.model.setRootPath('')
        self.tree = QTreeView(); self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed|QAbstractItemView.SelectedClicked)
        self.tree.setColumnHidden(1,True); self.tree.setColumnHidden(3,True)
        hdr = self.tree.header(); hdr.setSectionResizeMode(0,QHeaderView.Interactive); hdr.setSectionResizeMode(2,QHeaderView.Interactive)
        self.tree.clicked.connect(self.on_tree_clicked)
        l.addWidget(self.tree)  # File tree
        # Section: BIDStatistics
        l.addWidget(QLabel('<b>BIDStatistics</b>'))
        self.stats = QTreeWidget(); self.stats.setHeaderLabels(["Metric","Value"])
        self.stats.setAlternatingRowColors(True)
        s_hdr = self.stats.header(); s_hdr.setSectionResizeMode(0,QHeaderView.Interactive); s_hdr.setSectionResizeMode(1,QHeaderView.Interactive)
        l.addWidget(self.stats)
        splitter.addWidget(left)
        self.viewer = MetadataViewer(); splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1,2); self.setCentralWidget(splitter); self.resize(1400,900)
        self.tree.setRootIndex(self.model.index(''))

    def open_dir(self):
        p = QFileDialog.getExistingDirectory(self,"Select BIDS dataset")
        if p:
            self.bids_root = Path(p)
            self.model.setRootPath(p)
            self.tree.setRootIndex(self.model.index(p))
            self.viewer.clear()
            self.update_stats()
            self.setWindowTitle(f"BIDS Editor – {p}")

    def on_tree_clicked(self, idx: QModelIndex):
        p = Path(self.model.filePath(idx))
        self.selected = p
        if p.suffix.lower() in ['.json','.tsv']:
            self.viewer.load_file(p)

    def update_stats(self):
        root = self.bids_root
        self.stats.clear()
        subs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total subjects",str(len(subs))]))
        files = list(root.rglob('*.*'))
        self.stats.addTopLevelItem(QTreeWidgetItem(["Total files",str(len(files))]))
        for sub in subs:
            si = QTreeWidgetItem([sub.name,""])
            sessions = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith('ses-')]
            if len(sessions)>1:
                for ses in sessions:
                    s2 = QTreeWidgetItem([ses.name,""])
                    mods = set(p.parent.name for p in ses.rglob('*.nii*'))
                    s2.addChild(QTreeWidgetItem(["Modalities",str(len(mods))]))
                    for m in mods:
                        imgs = len(list(ses.rglob(f'{m}/*.*')))
                        meta = len(list(ses.rglob(f'{m}/*.json'))) + len(list(ses.rglob(f'{m}/*.tsv')))
                        s2.addChild(QTreeWidgetItem([m,f"imgs:{imgs}, meta:{meta}"]))
                    si.addChild(s2)
            else:
                mods = set(p.parent.name for p in sub.rglob('*.nii*'))
                si.addChild(QTreeWidgetItem(["Sessions","1"]))
                si.addChild(QTreeWidgetItem(["Modalities",str(len(mods))]))
                for m in mods:
                    imgs = len(list(sub.rglob(f'{m}/*.*')))
                    meta = len(list(sub.rglob(f'{m}/*.json'))) + len(list(sub.rglob(f'{m}/*.tsv')))
                    si.addChild(QTreeWidgetItem([m,f"imgs:{imgs}, meta:{meta}"]))
            self.stats.addTopLevelItem(si)
        self.stats.expandAll()

    def batch_rename(self):
        # Open the advanced remap dialog
        dlg = RemapDialog(self, getattr(self, 'selected', self.bids_root))
        dlg.exec_()

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette(); pal.setColor(QPalette.Window, QColor(240,240,240))
    pal.setColor(QPalette.WindowText, Qt.black)
    app.setPalette(pal)
    win = MainWindow(); win.show(); sys.exit(app.exec_())


if __name__ == '__main__':
    main()

