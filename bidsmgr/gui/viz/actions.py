"""The Qt side of actions: QActions, shortcuts, toolbar buttons.

:class:`ActionManager` turns every :class:`~bidsmgr.viz.actions.ActionDef`
into one ``QAction`` attached to the viewer, with its shortcut taken from the
user's keymap and scoped to the viewer and its children
(``WidgetWithChildrenShortcut``), so typing an "a" in a text field elsewhere
never switches the plane. Buttons, menus and the help are built from those
same QActions, so they always agree with the keys.

Enabled and checked states are recomputed from a context dict the presenter
supplies, after every scene change.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from PyQt6.QtCore import QObject, QSize, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QPushButton, QWidget

from ...viz import actions as A
from ...viz import keynames

log = logging.getLogger(__name__)


def qt_key(key: str) -> QKeySequence:
    """Our key spelling to a QKeySequence (``PgUp`` -> ``PgUp``)."""
    return QKeySequence(key)


class ActionManager(QObject):
    """Every action of one viewer, as QActions."""

    def __init__(self, host: QWidget, run_command: Callable[..., Any],
                 run_gui: Callable[[str, Mapping[str, Any]], None]) -> None:
        super().__init__(host)
        self.host = host
        self._run_command = run_command
        self._run_gui = run_gui
        self.actions: dict[str, QAction] = {}
        self._defs: dict[str, A.ActionDef] = {}
        self._ctx: dict[str, Any] = {}
        for d in A.ACTIONS:
            act = QAction(d.title, host)
            act.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            act.setCheckable(d.checkable)
            tip = d.help or d.title
            act.setToolTip(tip)
            act.triggered.connect(lambda _checked=False, a=d: self.trigger(a.id))
            host.addAction(act)
            self.actions[d.id] = act
            self._defs[d.id] = d
        #: Buttons with an icon, re-coloured when the theme changes.
        self._icon_buttons: list = []
        self.refresh_icons()
        from .bridge import ThemeHub, connect_while_alive

        connect_while_alive(ThemeHub.instance().changed, self, lambda m, _t: m.refresh_icons())
        self.apply_keymap({})

    def track_icon(self, btn, name: str) -> None:
        """Keep ``btn``'s icon ``name`` in the theme's colours (a button
        that is not an action's: a menu button, the layout button)."""
        self._icon_buttons = [(b, n) for b, n in self._icon_buttons if b is not btn]
        self._icon_buttons.append((btn, name))

    def refresh_icons(self) -> None:
        """Every action's icon (and its buttons') in the current theme's
        colours: an icon drawn for the dark theme vanishes on the light."""
        from .. import icons

        for action_id, act in self.actions.items():
            name = self._defs[action_id].icon
            if name:
                act.setIcon(icons.icon(name))
        alive = []
        for btn, name in self._icon_buttons:
            try:
                btn.setIcon(icons.icon(name))
            except RuntimeError:        # deleted with its toolbar
                continue
            alive.append((btn, name))
        self._icon_buttons = alive

    # ------------------------------------------------------------------
    def apply_keymap(self, overrides: Mapping[str, list[str]]) -> None:
        keymap = A.effective_keymap(overrides)
        for action_id, keys in keymap.items():
            act = self.actions.get(action_id)
            if act is None:
                continue
            act.setShortcuts([qt_key(k) for k in keys])
            d = self._defs[action_id]
            tip = d.help or d.title
            if keys:
                tip = f"{tip}.  Shortcut: {', '.join(keynames.keys(keys))}"
            act.setToolTip(tip)

    def keys_for(self, action_id: str) -> list[str]:
        """The action's keys in the STORED spelling (``Ctrl+G``): data, for
        comparing and saving. To show one, ``key_labels``."""
        act = self.actions.get(action_id)
        if act is None:
            return []
        return [s.toString() for s in act.shortcuts()]

    def key_labels(self, action_id: str) -> list[str]:
        """The action's keys as this OS writes them (``⌘G`` on a Mac)."""
        return keynames.keys(self.keys_for(action_id))

    def trigger(self, action_id: str) -> None:
        d = self._defs.get(action_id)
        if d is None:
            return
        if not A.evaluate(d.when, self._ctx):
            return
        try:
            if d.command.startswith("gui:"):
                self._run_gui(d.command[4:], d.params)
            else:
                self._run_command(d.command, **dict(d.params))
        except Exception:  # noqa: BLE001 - an action must never crash the event loop
            log.exception("action %s failed", action_id)

    def update_state(self, ctx: Mapping[str, Any]) -> None:
        """Recompute enabled/checked from the presenter's context."""
        self._ctx = dict(ctx)
        for action_id, act in self.actions.items():
            d = self._defs[action_id]
            enabled = A.evaluate(d.when, self._ctx)
            if act.isEnabled() != enabled:
                act.setEnabled(enabled)
            shown = A.evaluate(d.shown, self._ctx)
            if act.isVisible() != shown:
                act.setVisible(shown)
            if d.checkable:
                on = A.evaluate(d.checked, self._ctx)
                if act.isChecked() != on:
                    # setChecked emits toggled and changed, never triggered,
                    # so no command runs; the buttons mirror it via changed.
                    act.setChecked(on)

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)

    def definition(self, action_id: str) -> Optional[A.ActionDef]:
        return self._defs.get(action_id)

    # ------------------------------------------------------------------
    def button(self, action_id: str, parent: Optional[QWidget] = None) -> QPushButton:
        """A toolbar pill that mirrors and triggers one action."""
        act = self.actions[action_id]
        d = self._defs[action_id]
        btn = QPushButton(d.label, parent)
        btn.setObjectName("tb-btn-toggle")
        btn.setCheckable(d.checkable)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if d.icon:
            from .. import icons

            btn.setIcon(icons.icon(d.icon))
            btn.setIconSize(QSize(16, 16))
            self._icon_buttons.append((btn, d.icon))

        def sync() -> None:
            btn.setEnabled(act.isEnabled())
            # Visibility is mirrored only when it CHANGES. setVisible(True)
            # on a button not yet in a layout shows it as a top-level window
            # of its own, which cost 2.5 ms a button when a viewer was built.
            if not act.isVisible():
                btn.setVisible(False)
            elif btn.isHidden() and btn.testAttribute(Qt.WidgetAttribute.WA_WState_ExplicitShowHide):
                btn.setVisible(True)
            btn.setToolTip(act.toolTip())
            if d.checkable and btn.isChecked() != act.isChecked():
                btn.blockSignals(True)
                btn.setChecked(act.isChecked())
                btn.blockSignals(False)

        act.changed.connect(sync)
        btn.clicked.connect(lambda _c=False: (self.trigger(action_id), sync()))
        sync()
        btn.setProperty("viz_action", action_id)
        return btn


__all__ = ["ActionManager", "qt_key"]
