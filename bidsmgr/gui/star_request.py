"""A request for a star on GitHub.

BIDS Manager collects no usage data, so a star is the one sign its authors
get that it is useful to someone. This dialog says so, says how (one click on
the project's page) and why it helps (others find the project; the user can
follow its updates).

Beside the text, an animation. The seven kinds of data appear, each in its
own shape, as on the documentation's home page: a brain (from the MNI
template BIDS Manager ships for its quality check), a spectrum with its
metabolites, a PET phantom, six EEG channels, a turning MEG helmet, the ECG
and the breathing, and a scanpath on a screen. They gather into the lines of
the BIDS Manager logo, the logo becomes a star, still dim, and a small
scientist walks up to a button and presses it: the star lights up.

When it asks (:func:`due`): at most once a day, on the first launch of a day
(the day is the computer's date, compared with the date of the last request),
right after What's new when both are due, until the user has gone to GitHub
from it. Opening it by hand counts as that day's request.
"Don't ask me again" here and Settings > Display turn it off; Home's "Help
us" button and the What's new dialog open it on request. Called from
:func:`bidsmgr.main.main`, never from the window, so a test that builds a
window gets no dialog.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, QUrl
from PyQt6.QtGui import (
    QColor,
    QDesktopServices,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .whats_new import KINDS, ease, logo_points, seven_colours

REPO_URL = "https://github.com/ANCPLabOldenburg/BIDS-Manager"

_TEMPLATES = Path(__file__).resolve().parent.parent / "qc" / "templates"

#: Particles per kind, in ``KINDS`` order: MRI, MR spectroscopy, PET, EEG,
#: MEG, physiology, eye tracking. The brain needs the most to read as one.
COUNTS = (400, 170, 220, 300, 240, 170, 170)

# The timeline, in seconds.
APPEAR = 0.5                  # each figure grows out of its place...
APPEAR_STAGGER = 0.12         # ...one after the other
GATHER_AT = 2.4               # the figures stream into the logo's lines
GATHER = 1.4
GATHER_STAGGER = 0.5
MORPH_AT = 4.8                # the logo becomes a star, dim
MORPH = 1.2
MORPH_STAGGER = 0.3
SCENE_AT = 5.7                # the ground and the button appear
WALK_AT = 6.0                 # the scientist walks in
WALK = 2.2
REACH = 0.25
PRESS_AT = WALK_AT + WALK + REACH
LIGHT = 0.9                   # the star lights up
CHEER_AT = PRESS_AT + 0.45   # three jumps, arms up
SETTLED = CHEER_AT + 1.85
GLINT_EVERY = 3.5             # once it shines, a glint crosses it this often
GLINT = 0.7

# Places, in the canvas' unit frame (x right, y down; the canvas is at least
# 2.3 units wide and 2 high).
RING = (0.9, 0.7)             # where the figures stand
FIGURE_HALF = 0.22
LOGO_SIZE = 1.35
LOGO_CENTRE = (0.0, -0.08)
STAR_CENTRE = (0.0, -0.32)
STAR_RADIUS = 0.56
#: How far the glow, rays and twinkles reach from the star's centre: inside
#: the canvas (the star's centre is 0.68 below its top edge).
SHINE_REACH = 0.66

# The scientist and the button.
GROUND = 0.9
FOOT_H = 0.008                # the ankle above the ground: the shoe
THIGH = SHIN = 0.08
LEG = THIGH + SHIN
TORSO = 0.16
UPPER = FORE = 0.085          # upper arm, forearm
ARM = UPPER + FORE
PRESS_REACH = 0.16            # shoulder to hand when pressing: the elbow just bent
HEAD_R = 0.066
NECK = 0.014
SHOULDER_DX = 0.01            # the shoulders sit a little forward of the hips
BUTTON_X = 0.78               # the console's middle
BUTTON_HALF = 0.075
CAP_Y = 0.68                  # the console's top, where the cap sits
CAP_RX, CAP_RY = 0.042, 0.022
PRESS_DEPTH = 0.012
#: Where the hand presses: the near side of the cap's top.
PUSH_AT = (BUTTON_X - 0.02, CAP_Y - 0.8 * CAP_RY)
START_X = -1.45
#: The hips' height standing: the knees a little bent.
HIP_Y = GROUND - FOOT_H - 0.94 * LEG
_SHOULDER_Y = HIP_Y - TORSO
#: Where the scientist stops: ``PRESS_REACH`` from where the hand presses.
STOP_X = (PUSH_AT[0] - float(np.sqrt(PRESS_REACH ** 2 - (PUSH_AT[1] - _SHOULDER_Y) ** 2))
          - SHOULDER_DX)
#: The celebration once the star lights: (start after ``CHEER_AT``, height).
JUMPS = ((0.0, 0.07), (0.45, 0.095), (0.9, 0.065))
JUMP = 0.42


# ---------------------------------------------------------------------------
# When to ask
# ---------------------------------------------------------------------------


@dataclass
class StarSchedule:
    """What the app remembers to decide when to ask (not a visible setting)."""

    last_asked: str = ""     # ISO date of the last request; "" = never asked
    opened: bool = False     # the user went to GitHub from the dialog


def due(schedule: StarSchedule, today: date, *, enabled: bool = True) -> bool:
    """Whether to ask today: once a day at most, until the user has starred."""
    if not enabled or schedule.opened:
        return False
    return schedule.last_asked != today.isoformat()


def _load() -> StarSchedule:
    from .app_settings import AppSettings

    return StarSchedule(**AppSettings.star_schedule())


def _save(schedule: StarSchedule) -> None:
    from .app_settings import AppSettings

    AppSettings.remember_star_schedule(**schedule.__dict__)


def maybe_show(window: QWidget, today: Optional[date] = None) -> Optional["StarDialog"]:
    """Ask over ``window`` if no request was shown yet today."""
    from .app_settings import AppSettings

    today = today or date.today()
    if not due(_load(), today, enabled=AppSettings.load().ask_for_star):
        return None
    return show_dialog(window, asked=True, today=today)


def show_dialog(parent: QWidget, *, asked: bool = False,
                today: Optional[date] = None) -> "StarDialog":
    """Open the dialog over ``parent`` (the daily request, Home's Help us
    button or the What's new dialog); it is deleted when closed. The day is
    remembered as soon as it opens, however it was opened, so it is never
    asked twice in a day."""
    schedule = _load()
    schedule.last_asked = (today or date.today()).isoformat()
    _save(schedule)
    dialog = StarDialog(parent, asked=asked)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()
    return dialog


# ---------------------------------------------------------------------------
# The seven forms, each in a [-1, 1] box (x right, y down; z for 3-D)
# ---------------------------------------------------------------------------


def _unit(pts: np.ndarray) -> np.ndarray:
    """Centre ``pts`` and scale them so the larger of x and y spans -1..1."""
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    scale = float((hi - lo).max()) / 2 or 1.0
    out = pts.astype(float).copy()
    out[:, :2] = (pts[:, :2] - (lo + hi) / 2) / scale
    if pts.shape[1] > 2:
        out[:, 2] = (pts[:, 2] - pts[:, 2].mean()) / scale
    return out


def _along(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points spread evenly along the curve (x, y)."""
    step = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(step)])
    at = np.sort(rng.random(n)) * s[-1]
    return np.stack([np.interp(at, s, x), np.interp(at, s, y)], axis=1)


@lru_cache(maxsize=None)
def _template(name: str) -> np.ndarray:
    """A volume of the MNI template the quality check ships (uint8 NIfTI-1).

    Read here rather than through nibabel: this dialog opens at start-up, on
    the GUI thread, and importing nibabel costs about 180 ms.
    """
    raw = gzip.decompress((_TEMPLATES / name).read_bytes())
    header = np.frombuffer(raw, "<i4", 1, 0)[0]
    dims = np.frombuffer(raw, "<i2", 8, 40)
    datatype = int(np.frombuffer(raw, "<i2", 1, 70)[0])
    offset = int(np.frombuffer(raw, "<f4", 1, 108)[0])
    if header != 348 or datatype != 2 or dims[0] < 3:
        raise ValueError(f"{name}: not a little-endian uint8 NIfTI-1 volume")
    shape = tuple(int(d) for d in dims[1:4])
    return np.frombuffer(raw, np.uint8, int(np.prod(shape)), offset).reshape(shape, order="F")


def _brain(n: int, rng: np.random.Generator) -> np.ndarray:
    """A brain seen from the side, nose to the left, in 3-D: the template's
    surface and grey matter."""
    try:
        t1 = _template("mni2009c_2mm_T1w.nii.gz") / 255.0
        mask = _template("mni2009c_2mm_brain_mask.nii.gz") > 127
    except (OSError, ValueError):     # no template: an ellipsoid stands in
        u = rng.standard_normal((n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return u * np.array([1.0, 0.8, 0.7])
    inner = mask.copy()
    inner[1:-1, 1:-1, 1:-1] &= (mask[:-2, 1:-1, 1:-1] & mask[2:, 1:-1, 1:-1]
                                & mask[1:-1, :-2, 1:-1] & mask[1:-1, 2:, 1:-1]
                                & mask[1:-1, 1:-1, :-2] & mask[1:-1, 1:-1, 2:])
    surface = mask & ~inner
    grey = mask & (t1 > 0.35) & (t1 < 0.72)
    weight = (3.0 * surface + grey).ravel()
    pick = rng.choice(weight.size, size=n, p=weight / weight.sum())
    i, j, k = np.unravel_index(pick, mask.shape)
    vox = np.stack([i, j, k], axis=1) + rng.random((n, 3))
    # Voxel axes (left to right, back to front, down to up) to the page:
    # the front to the left, up to the top, left to right towards the viewer.
    return _unit(np.stack([-vox[:, 1], -vox[:, 2], vox[:, 0]], axis=1))


def _spectrum(n: int, rng: np.random.Generator) -> np.ndarray:
    """An MR spectrum, ppm from right to left, with dotted markers at the
    metabolites BIDS Manager's viewer labels (NAA, Cr, Cho, mI)."""
    peaks = ((2.01, 1.0, 0.03), (3.03, 0.55, 0.03), (3.22, 0.5, 0.03), (3.56, 0.3, 0.04),
             (2.25, 0.15, 0.06), (2.35, 0.12, 0.05), (3.75, 0.14, 0.05),
             (0.9, 0.1, 0.06), (1.3, 0.12, 0.07))
    ppm = np.linspace(4.1, 0.6, 1400)
    height = sum(h * np.exp(-((ppm - c) / w) ** 2) for c, h, w in peaks)
    x = (2.35 - ppm) / 1.75
    y = 0.55 - 1.15 * height
    curve = _along(x, y, int(n * 0.84), rng)
    marks = []
    per = (n - len(curve)) // 4
    for c, h, _w in peaks[:4]:
        mx = (2.35 - c) / 1.75
        top = 0.55 - 1.15 * h + 0.07
        marks.append(np.stack([np.full(per, mx), np.linspace(0.55, top, per)], axis=1))
    pts = np.concatenate([curve, *marks])
    extra = n - len(pts)
    return np.concatenate([pts, curve[rng.integers(0, len(curve), extra)]])


def _rounded_outline(half_w: float, half_h: float, r_top: float, r_bottom: float) -> np.ndarray:
    """A closed outline: a rectangle with round top corners and nearly square
    bottom ones (the body of a NEMA phantom, axially)."""
    def arc(cx, cy, r, a0, a1):
        a = np.linspace(a0, a1, 24)
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    return np.concatenate([
        arc(half_w - r_top, -half_h + r_top, r_top, -np.pi / 2, 0),
        arc(half_w - r_bottom, half_h - r_bottom, r_bottom, 0, np.pi / 2),
        arc(-half_w + r_bottom, half_h - r_bottom, r_bottom, np.pi / 2, np.pi),
        arc(-half_w + r_top, -half_h + r_top, r_top, np.pi, 1.5 * np.pi),
    ])


def _phantom(n: int, rng: np.random.Generator) -> np.ndarray:
    """A PET image of the NEMA body phantom: a body of background activity,
    a cold lung insert in the middle, and six hot spheres around it."""
    outline = _rounded_outline(1.0, 0.72, 0.55, 0.12)
    edge = _along(np.append(outline[:, 0], outline[0, 0]),
                  np.append(outline[:, 1], outline[0, 1]), int(n * 0.16), rng)
    centre = np.array([0.0, 0.08])
    radii = np.array([10, 13, 17, 22, 28, 37]) / 37 * 0.17
    angles = np.deg2rad(-150 + 60 * np.arange(6))
    spheres_at = centre + 0.44 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    hot = int(n * 0.6)
    which = rng.choice(6, size=hot, p=radii ** 2 / (radii ** 2).sum())
    a, r = 2 * np.pi * rng.random(hot), np.sqrt(rng.random(hot))
    spheres = spheres_at[which] + (radii[which] * r)[:, None] * np.stack([np.cos(a), np.sin(a)], 1)
    # Background: inside the body, outside the lung insert and the spheres.
    rest = n - len(edge) - hot
    cand = rng.uniform([-0.95, -0.68], [0.95, 0.68], (rest * 8, 2))
    corner = (cand[:, 1] < -0.17) & (np.abs(cand[:, 0]) > 0.45) \
        & (np.hypot(np.abs(cand[:, 0]) - 0.45, cand[:, 1] + 0.17) > 0.53)
    keep = ~corner & (np.hypot(*(cand - centre).T) > 0.16)
    for s, rad in zip(spheres_at, radii):
        keep &= np.hypot(*(cand - s).T) > rad
    background = cand[keep][:rest]
    pts = np.concatenate([edge, spheres, background])
    if len(pts) < n:                    # top up from the spheres if short
        pts = np.concatenate([pts, spheres[rng.integers(0, hot, n - len(pts))]])
    return pts


def _eeg(n: int, rng: np.random.Generator) -> np.ndarray:
    """Six EEG channels: x along the trace, y the channel (scrolled later)."""
    lanes = rng.integers(0, 6, n)
    return np.stack([rng.uniform(-1, 1, n), lanes.astype(float)], axis=1)


def _eeg_y(x: np.ndarray, lane: np.ndarray, t: float) -> np.ndarray:
    s = x + 0.3 * t
    ph = lane * 1.7
    alpha = np.where(lane >= 4, 0.45, 0.15)       # the back of the head: alpha
    wave = (0.5 * np.sin(2 * np.pi * 2.6 * s + ph) + alpha * np.sin(2 * np.pi * 7.5 * s + 2 * ph)
            + 0.22 * np.sin(2 * np.pi * 13.0 * s + 3 * ph) + 0.3 * np.sin(2 * np.pi * 0.9 * s + ph))
    return -0.8 + lane * (1.6 / 5) + 0.11 * wave


def _physio(n: int, rng: np.random.Generator) -> np.ndarray:
    """The ECG above, the breathing below: x along the trace, y the lane."""
    lanes = (rng.random(n) < 0.42).astype(float)    # 0 = ECG, 1 = breathing
    return np.stack([rng.uniform(-1, 1, n), lanes], axis=1)


def _physio_y(x: np.ndarray, lane: np.ndarray, t: float) -> np.ndarray:
    s = x + 0.25 * t
    beat = np.mod(s / 0.55, 1.0)
    ecg = (0.12 * np.exp(-((beat - 0.18) / 0.035) ** 2) - 0.08 * np.exp(-((beat - 0.36) / 0.012) ** 2)
           + 0.75 * np.exp(-((beat - 0.4) / 0.013) ** 2) - 0.18 * np.exp(-((beat - 0.43) / 0.014) ** 2)
           + 0.22 * np.exp(-((beat - 0.65) / 0.06) ** 2))
    breath = np.sin(2 * np.pi * s / 1.3) * (0.8 + 0.2 * np.sin(2 * np.pi * s / 3.1))
    return np.where(lane == 0, -0.3 - 0.6 * ecg, 0.55 - 0.22 * breath)


def _scanpath(n: int, rng: np.random.Generator) -> np.ndarray:
    """Eye tracking: fixations sized by how long they last, joined by
    saccades, on the outline of the screen."""
    fix = np.array([[-0.62, -0.18], [-0.2, -0.32], [0.18, -0.12], [0.55, -0.3],
                    [0.64, 0.18], [0.2, 0.3], [-0.25, 0.12], [-0.6, 0.3]])
    dur = np.array([0.6, 0.3, 0.9, 0.4, 0.5, 0.8, 0.35, 0.45])
    k_fix = int(n * 0.5)
    which = rng.choice(len(fix), size=k_fix, p=dur / dur.sum())
    rad = 0.035 + 0.06 * np.sqrt(dur[which])
    a, r = 2 * np.pi * rng.random(k_fix), np.sqrt(rng.random(k_fix))
    blobs = fix[which] + (rad * r)[:, None] * np.stack([np.cos(a), np.sin(a)], axis=1)
    path = _along(fix[:, 0], fix[:, 1], int(n * 0.3), rng)
    rest = n - k_fix - len(path)
    frame = np.array([[-1, -0.5625], [1, -0.5625], [1, 0.5625], [-1, 0.5625], [-1, -0.5625]])
    screen = _along(frame[:, 0], frame[:, 1], rest, rng)
    return np.concatenate([blobs, path, screen])


def _helmet(n: int, rng: np.random.Generator) -> np.ndarray:
    """An MEG helmet in 3-D: its sensors as small squares on the cap, and a
    faint surface between them."""
    sensors = 36
    k = np.arange(sensors) + 0.5
    z = 1 - (k / sensors) * 1.25                    # from the top to below the equator
    r = np.sqrt(np.clip(1 - z ** 2, 0, None))
    phi = k * np.pi * (3 - np.sqrt(5))
    centres = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    per = int(n * 0.8) // sensors
    squares = []
    for c in centres:
        a = np.cross(c, [0.0, 0.0, 1.0])
        a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])
        b = np.cross(c, a)
        u = rng.random(per) * 4
        side, f = u.astype(int), u - u.astype(int)
        h = 0.07
        s = np.choose(side, [-h + 2 * h * f, np.full(per, h), h - 2 * h * f, np.full(per, -h)])
        w = np.choose(side, [np.full(per, -h), -h + 2 * h * f, np.full(per, h), h - 2 * h * f])
        squares.append(c + s[:, None] * a + w[:, None] * b)
    pts = np.concatenate(squares)
    rest = n - len(pts)
    v = rng.standard_normal((rest, 3))
    v[:, 2] = np.abs(v[:, 2]) - 0.2
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = np.concatenate([pts, 0.97 * v])
    # Device axes (x right, y front, z up) to the page.
    return _unit(np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1))


_FORMS = (_brain, _spectrum, _phantom, _eeg, _helmet, _physio, _scanpath)


def form_points(kind: int, base: np.ndarray, t: float) -> np.ndarray:
    """The form of ``kind`` at time ``t``, in its [-1, 1] box: the brain and
    the helmet turn, the traces scroll, the rest hold still."""
    if kind in (0, 4):
        # The helmet turns; the brain rocks, so it stays a brain seen from the side.
        a = 0.55 * t if kind == 4 else 0.35 * np.sin(0.6 * t)
        x = base[:, 0] * np.cos(a) + base[:, 2] * np.sin(a)
        return np.stack([x, base[:, 1]], axis=1)
    if kind == 3:
        return np.stack([base[:, 0], _eeg_y(base[:, 0], base[:, 1], t)], axis=1)
    if kind == 5:
        return np.stack([base[:, 0], _physio_y(base[:, 0], base[:, 1], t)], axis=1)
    return base[:, :2]


# ---------------------------------------------------------------------------
# The star
# ---------------------------------------------------------------------------


def star_outline(radius: float = 1.0, inner: float = 0.42) -> np.ndarray:
    """The ten corners of a five-pointed star, the top point first, centred
    at the origin (y down, as on screen)."""
    angle = -np.pi / 2 + np.arange(10) * np.pi / 5
    r = np.where(np.arange(10) % 2 == 0, radius, radius * inner)
    return np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)


def _star_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points of a star of radius 1: mostly its lines, some inside."""
    corners = star_outline()
    edge_n = int(n * 0.65)
    edge = rng.integers(0, 10, edge_n)
    t = rng.random(edge_n)[:, None]
    on_edge = corners[edge] * (1 - t) + corners[(edge + 1) % 10] * t
    # Inside: the ten triangles from the centre to each edge have one area.
    tri = rng.integers(0, 10, n - edge_n)
    a, b = rng.random(n - edge_n), rng.random(n - edge_n)
    flip = a + b > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    inside = corners[tri] * a[:, None] + corners[(tri + 1) % 10] * b[:, None]
    return np.concatenate([on_edge, inside])


def _polar(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])


def _pair_by_angle(a: np.ndarray, b: np.ndarray, bins: int = 48) -> np.ndarray:
    """An order of ``b`` that pairs each point of ``a`` with a point of ``b``
    in the same direction from the centre, inner with inner, so a morph from
    ``a`` to ``b`` moves every particle a short way."""
    n = len(a)
    oa, ob = np.argsort(_polar(a)[1]), np.argsort(_polar(b)[1])
    out = np.empty(n, dtype=int)
    edges = np.linspace(0, n, bins + 1).astype(int)
    ra, rb = _polar(a)[0], _polar(b)[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        ia, ib = oa[lo:hi], ob[lo:hi]
        out[ia[np.argsort(ra[ia])]] = ib[np.argsort(rb[ib])]
    return out


def _bump(t: float, start: float, rise: float, fall: float) -> float:
    """0 before ``start``, up to 1 over ``rise``, back to 0 over ``fall``."""
    if t <= start:
        return 0.0
    if t < start + rise:
        return (t - start) / rise
    return max(0.0, 1.0 - (t - start - rise) / fall)


def lit(t: float) -> float:
    """How far the star has lit up: 0 until the button is pressed, then 1."""
    return float(ease(np.array((t - PRESS_AT) / LIGHT)))


def dimness(t: float) -> float:
    """How dim the star is: it forms dim, and the button lights it."""
    return float(ease(np.array((t - MORPH_AT) / MORPH))) * (1.0 - lit(t))


# ---------------------------------------------------------------------------
# The particles: where each one is, and its colour, at time t
# ---------------------------------------------------------------------------


class StarParticles:
    """The seven forms, the logo's lines, the star. Pure numbers (no Qt but
    the logo's pixels), so the timeline can be tested: :meth:`at` gives
    every particle's place at time ``t`` in the canvas' unit frame and
    :meth:`colours` its colour."""

    def __init__(self, counts: tuple[int, ...] = COUNTS, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        k = len(KINDS)
        self.n = int(sum(counts))
        self.kind = np.repeat(np.arange(k), counts)
        forms = []
        for make, count in zip(_FORMS, counts):
            base = make(count, rng)
            if base.shape[1] == 2:
                base = np.concatenate([base, np.zeros((count, 1))], axis=1)
            forms.append(base)
        self.base = np.concatenate(forms)
        self.members = [np.nonzero(self.kind == i)[0] for i in range(k)]
        angle = -np.pi / 2 + 2 * np.pi * np.arange(k) / k
        self.anchor = np.stack([RING[0] * np.cos(angle), RING[1] * np.sin(angle)], axis=1)
        self.size = np.where(np.arange(k) == 0, FIGURE_HALF * 1.15, FIGURE_HALF)
        self.appear = APPEAR_STAGGER * np.arange(k)
        # The logo's lines: each kind takes the part facing its figure.
        targets = LOGO_SIZE * logo_points(self.n, rng)
        order = np.argsort(np.mod(np.arctan2(targets[:, 1], targets[:, 0]) - angle[0], 2 * np.pi))
        self.logo = np.empty_like(targets)
        start = 0
        for i, count in enumerate(counts):
            self.logo[self.members[i]] = targets[order[start:start + count]]
            start += count
        self.logo += LOGO_CENTRE
        self.gather_delay = GATHER_STAGGER * rng.random(self.n)
        self.morph_delay = MORPH_STAGGER * rng.random(self.n)
        star = STAR_RADIUS * _star_points(self.n, rng)
        self.star = star[_pair_by_angle(self.logo - LOGO_CENTRE, star)] + STAR_CENTRE

    def figures_at(self, t: float) -> np.ndarray:
        out = np.empty((self.n, 2))
        for i, mine in enumerate(self.members):
            grow = float(ease(np.array((t - self.appear[i]) / APPEAR)))
            pts = form_points(i, self.base[mine], t)
            out[mine] = self.anchor[i] + self.size[i] * grow * pts
        return out

    def at(self, t: float) -> np.ndarray:
        start = self.figures_at(min(t, GATHER_AT + GATHER + GATHER_STAGGER))
        g = ease((t - GATHER_AT - self.gather_delay) / GATHER)[:, None]
        mid = (start + self.logo) / 2
        bend = np.stack([-(self.logo - start)[:, 1], (self.logo - start)[:, 0]], axis=1)
        ctrl = mid + 0.18 * bend
        pos = (1 - g) ** 2 * start + 2 * (1 - g) * g * ctrl + g ** 2 * self.logo
        m = ease((t - MORPH_AT - self.morph_delay) / MORPH)
        if m.any():
            lc, sc = np.array(LOGO_CENTRE), np.array(STAR_CENTRE)
            r1, a1 = _polar(self.logo - lc)
            r2, a2 = _polar(self.star - sc)
            turn = np.mod(a2 - a1 + np.pi, 2 * np.pi) - np.pi
            r = r1 + (r2 - r1) * m
            a = a1 + turn * m + 0.3 * np.sin(np.pi * m)
            centre = lc + (sc - lc) * m[:, None]
            morphed = centre + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
            pos = np.where((m > 0)[:, None], morphed, pos)
        # The star swells when it lights, then breathes.
        swell = 1.0 + 0.08 * _bump(t, PRESS_AT, 0.18, 0.5) + 0.012 * lit(t) * np.sin(1.4 * t)
        if swell != 1.0:
            centre = np.array(STAR_CENTRE)
            pos = centre + (pos - centre) * swell
        return pos

    def colours(self, t: float, kinds_rgb: np.ndarray, unlit_rgb: np.ndarray) -> np.ndarray:
        """Every particle's colour (0..1 RGB): its kind's, greyed towards
        ``unlit_rgb`` (the theme's muted text) while the star is dim, and a
        glint once it shines."""
        rgb = kinds_rgb[self.kind]
        rgb = rgb + (unlit_rgb - rgb) * (0.72 * dimness(t))
        cycle = (t - SETTLED) % GLINT_EVERY if t > SETTLED else GLINT_EVERY
        if cycle < GLINT:
            band = -1.1 + 2.2 * cycle / GLINT
            rel = (self.star - STAR_CENTRE) / STAR_RADIUS
            d = rel[:, 0] + 0.5 * rel[:, 1] - band
            light = 0.55 * np.exp(-(d / 0.16) ** 2)[:, None]
            rgb = rgb + (1 - rgb) * light
        return rgb

    @staticmethod
    def labels_alpha(t: float) -> float:
        if t < APPEAR:
            return t / APPEAR
        return float(np.clip(1.0 - (t - GATHER_AT) / 0.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# The scientist
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """Where the scientist is, in the unit frame: the hips, the ankles and the
    hands. The knees and elbows follow from them (:func:`joint`)."""

    x: float                                                 # the hips
    hip_y: float
    feet: tuple[tuple[float, float], tuple[float, float]]    # back, front
    hands: tuple[tuple[float, float], tuple[float, float]]   # back, front
    looks_up: bool
    smile: float                                             # 0..1


def shoulder(pose: Pose) -> tuple[float, float]:
    """The front shoulder (the back one is a little behind it)."""
    return pose.x + SHOULDER_DX, pose.hip_y - TORSO


def joint(a: tuple[float, float], b: tuple[float, float], l1: float, l2: float,
          bend: float) -> tuple[float, float]:
    """The middle joint of a limb from ``a`` to ``b`` with segments ``l1`` and
    ``l2``: ``bend`` +1 puts it behind the line (an elbow), -1 in front (a
    knee). A limb asked to reach too far is straight."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = float(np.hypot(dx, dy))
    if d < 1e-9:
        return a[0], a[1] + l1
    ux, uy = dx / d, dy / d
    d = min(max(d, abs(l1 - l2) + 1e-6), l1 + l2 - 1e-6)
    along = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    h = float(np.sqrt(max(l1 ** 2 - along ** 2, 0.0)))
    return a[0] + along * ux - bend * h * uy, a[1] + along * uy + bend * h * ux


def cap_offset(t: float) -> float:
    """How far the button is pushed in."""
    return PRESS_DEPTH * _bump(t, PRESS_AT, 0.12, 0.25)


def jump_at(t: float) -> tuple[float, float, float]:
    """The celebration at ``t``: how high the hips are lifted, how far they
    dip (before a jump and on landing), and how tucked the legs are."""
    for start, height in JUMPS:
        v = (t - CHEER_AT - start) / JUMP
        if not 0.0 <= v < 1.0:
            continue
        if v < 0.18:
            return 0.0, 0.025 * float(np.sin(np.pi * v / 0.18)), 0.0
        if v < 0.82:
            s = (v - 0.18) / 0.64
            return height * float(np.sin(np.pi * s)), 0.0, float(np.sin(np.pi * s))
        return 0.0, 0.025 * float(np.sin(np.pi * (v - 0.82) / 0.18)), 0.0
    return 0.0, 0.0, 0.0


def pose_at(t: float) -> Pose:
    """The scientist at time ``t``: walking in, reaching, pressing, then
    three jumps with arms up, then looking up at the star."""
    u = float(np.clip((t - WALK_AT) / WALK, 0.0, 1.0))
    x = START_X + (STOP_X - START_X) * (1 - (1 - u) ** 2)       # slows to a stop
    walk = (1 - u ** 4) if 0 < u < 1 else 0.0                   # how much of a stride
    phase = 2 * np.pi * 1.6 * (t - WALK_AT)
    lift, dip, tuck = jump_at(t)
    hip_y = HIP_Y - 0.008 * abs(float(np.sin(phase))) * walk + dip - lift

    def foot(ph: float, side: float) -> tuple[float, float]:
        if lift > 0:                                 # in the air: the legs tucked
            return x + side * 0.012 + 0.015 * tuck, hip_y + 0.94 * LEG - 0.05 * tuck
        return (x + side * 0.012 + 0.05 * walk * float(np.sin(ph)),
                GROUND - FOOT_H - 0.03 * walk * max(0.0, float(np.cos(ph))))

    feet = (foot(phase + np.pi, -1.0), foot(phase, 1.0))
    sx, sy = x + SHOULDER_DX, hip_y - TORSO
    bsx, bsy = sx - 0.022, sy + 0.004
    swing = 0.06 * walk * float(np.sin(phase))
    back = (bsx + swing - 0.01, bsy + 0.155)
    front = (sx - swing + 0.012, sy + 0.155)
    # Reach, press, let go.
    target = (PUSH_AT[0], PUSH_AT[1] + cap_offset(t))
    reach = float(ease(np.array((t - (WALK_AT + WALK)) / REACH)))
    release = float(ease(np.array((t - PRESS_AT - 0.2) / 0.25)))
    touch = reach * (1 - release)
    front = (front[0] + (target[0] - front[0]) * touch, front[1] + (target[1] - front[1]) * touch)
    # Arms up for the jumps.
    end = CHEER_AT + JUMPS[-1][0] + JUMP
    cheer = float(np.clip((t - CHEER_AT + 0.1) / 0.15, 0, 1)) * float(np.clip(1 - (t - end) / 0.3, 0, 1))
    if cheer > 0:
        # A wide V, clear of the head.
        up_front, up_back = (sx + 0.125, sy - 0.105), (bsx - 0.12, bsy - 0.105)
        front = (front[0] + (up_front[0] - front[0]) * cheer, front[1] + (up_front[1] - front[1]) * cheer)
        back = (back[0] + (up_back[0] - back[0]) * cheer, back[1] + (up_back[1] - back[1]) * cheer)
    return Pose(x=float(x), hip_y=float(hip_y),
                feet=tuple((float(a), float(b)) for a, b in feet),
                hands=((float(back[0]), float(back[1])), (float(front[0]), float(front[1]))),
                looks_up=t >= PRESS_AT, smile=float(np.clip((t - PRESS_AT) / 0.4, 0.0, 1.0)))


def scene_alpha(t: float) -> float:
    return float(np.clip((t - SCENE_AT) / 0.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

#: Particle colours are rounded to this many steps a channel, one drawPoints
#: per colour.
_LEVELS = 12


def _qcolor(rgb: np.ndarray, alpha: float = 1.0) -> QColor:
    colour = QColor.fromRgbF(*(float(v) for v in np.clip(rgb, 0, 1)))
    colour.setAlphaF(float(np.clip(alpha, 0, 1)))
    return colour


def _with_alpha(colour: str, alpha: float) -> QColor:
    out = QColor(colour)
    out.setAlphaF(float(np.clip(alpha, 0, 1)))
    return out


class StarCanvas(QWidget):
    """The animation: the seven kinds of data become the BIDS Manager logo,
    the logo a dim star, and a scientist's button lights it."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts
        from .viz.bridge import ThemeHub, connect_while_alive

        self.setObjectName("star-canvas")
        self.setAccessibleName(
            "Animation: MRI, MR spectroscopy, PET, EEG, MEG, physiology and eye "
            "tracking become the BIDS Manager logo, the logo becomes a dim star, and "
            "a scientist presses a button that lights it up")
        self.setMinimumSize(fonts.px(300), fonts.px(320))
        self.particles = StarParticles()
        self._clock = QElapsedTimer()
        self._frozen: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        self._paint_ms = 0.0
        self._kinds_key = None
        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w.update())

    def set_time(self, t: Optional[float]) -> None:
        """Freeze the animation at ``t`` seconds (``None`` lets it run)."""
        self._frozen = t
        self.update()

    def time(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return self._clock.elapsed() / 1000.0 if self._clock.isValid() else 0.0

    def is_running(self) -> bool:
        return self._timer.isActive()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt
        super().showEvent(event)
        if not self._clock.isValid():
            self._clock.start()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt
        self._timer.stop()
        super().hideEvent(event)

    def _kinds_rgb(self, theme) -> np.ndarray:
        """The seven kinds' colours in this theme, picked once per palette."""
        tokens = {**theme.tokens, **{f"series{i + 1}": theme.series(i) for i in range(6)}}
        key = tuple(sorted(tokens.items()))
        if self._kinds_key != key:
            self._kinds_key = key
            self._kinds = np.array([QColor(c).getRgbF()[:3] for c in seven_colours(tokens)])
        return self._kinds

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt
        from ..viz.theme import VizTheme
        from .theme_manager import CUR
        from .viz import fonts

        palette = CUR()
        theme = VizTheme.from_palette(palette)
        clock = QElapsedTimer()
        clock.start()
        t = self.time()
        if self._frozen is None and (t > SETTLED + 2 or self._paint_ms > 12.0) \
                and self._timer.interval() == 16:
            self._timer.setInterval(33)          # only the shine and a glint move now
        w, h = self.width(), self.height()
        unit = min(w / 2.3, h / 2.0)
        cx, cy = w / 2, h / 2

        def at(x: float, y: float) -> QPointF:
            return QPointF(cx + unit * x, cy + unit * y)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        parts = self.particles
        shine = lit(t)
        if shine > 0:
            self._paint_shine(p, at, unit, t, shine, palette)

        unlit = np.array(QColor(palette.get("muted", theme.dim)).getRgbF()[:3])
        rgb = parts.colours(t, self._kinds_rgb(theme), unlit)
        pos = parts.at(t)
        # One drawPoints per colour: rounded to 12 steps a channel, the
        # particles sorted by colour and cut where it changes.
        steps = np.rint(np.clip(rgb, 0, 1) * (_LEVELS - 1)).astype(int)
        keys = (steps[:, 0] * _LEVELS + steps[:, 1]) * _LEVELS + steps[:, 2]
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        points = list(map(QPointF, (cx + unit * pos[order, 0]).tolist(),
                          (cy + unit * pos[order, 1]).tolist()))
        cuts = np.flatnonzero(np.diff(keys)) + 1
        dot = max(2.4, fonts.px(2) * 1.6)
        for a, b in zip([0, *cuts.tolist()], [*cuts.tolist(), len(keys)]):
            key = int(keys[a])
            level = np.array([key // (_LEVELS * _LEVELS), (key // _LEVELS) % _LEVELS,
                              key % _LEVELS]) / (_LEVELS - 1)
            p.setPen(QPen(_qcolor(level, 0.92), dot, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            p.drawPoints(points[a:b])

        label_alpha = parts.labels_alpha(t)
        if label_alpha > 0:
            p.setFont(fonts.font(fonts.LABEL_PX))
            p.setPen(_with_alpha(theme.dim, label_alpha))
            box_w, box_h = unit * 0.7, fonts.px(16)
            for i, name in enumerate(KINDS):
                ax, ay = parts.anchor[i]
                top = ay + parts.size[i] * 1.02 + 0.02
                p.drawText(QRectF(cx + unit * ax - box_w / 2, cy + unit * top, box_w, box_h),
                           Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, name.upper())

        scene = scene_alpha(t)
        if scene > 0:
            self._paint_scene(p, at, unit, t, scene, palette, fonts)
        p.end()
        self._paint_ms = 0.9 * self._paint_ms + 0.1 * clock.nsecsElapsed() / 1e6

    @staticmethod
    def _paint_shine(p: QPainter, at, unit: float, t: float, shine: float, palette: dict) -> None:
        """Once lit: a glow and slow rays around the star, and twinkles, in
        the theme's accent."""
        from .viz import fonts

        accent = palette.get("accent", "#58a6ff")
        centre = at(*STAR_CENTRE)
        reach = unit * SHINE_REACH
        pulse = shine * (0.85 + 0.15 * np.sin(2.0 * t))
        glow = QRadialGradient(centre, reach)
        glow.setColorAt(0.0, _with_alpha(accent, 0.24 * pulse))
        glow.setColorAt(0.55, _with_alpha(accent, 0.08 * pulse))
        glow.setColorAt(1.0, _with_alpha(accent, 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(centre, reach, reach)
        near = unit * STAR_RADIUS * 0.5
        for far, half, offset in ((reach, 0.03, 0.0), (reach * 0.8, 0.018, np.pi / 8)):
            path = QPainterPath()
            for i in range(8):
                a = 0.12 * t + offset + 2 * np.pi * i / 8
                path.addPolygon(QPolygonF([
                    QPointF(centre.x() + near * np.cos(a), centre.y() + near * np.sin(a)),
                    QPointF(centre.x() + far * np.cos(a - half), centre.y() + far * np.sin(a - half)),
                    QPointF(centre.x() + far * np.cos(a + half), centre.y() + far * np.sin(a + half)),
                ]))
            brush = QRadialGradient(centre, far)
            brush.setColorAt(0.0, _with_alpha(accent, 0.0))
            brush.setColorAt(0.6, _with_alpha(accent, 0.16 * shine))
            brush.setColorAt(1.0, _with_alpha(accent, 0.0))
            p.fillPath(path, brush)
        for i in range(8):
            angle = -np.pi / 2 + 2 * np.pi * (i + 0.5) / 8
            dist = unit * (STAR_RADIUS + (SHINE_REACH - STAR_RADIUS) * (0.3 + 0.6 * ((i * 7) % 5) / 4))
            x = centre.x() + dist * np.cos(angle)
            y = centre.y() + dist * np.sin(angle)
            size = fonts.px(8) * max(0.0, np.sin(2.3 * t + 1.7 * i)) ** 3
            if size < 0.5:
                continue
            thin = 0.2 * size
            p.setBrush(_with_alpha(accent, 0.95 * shine))
            p.drawPolygon(QPolygonF([
                QPointF(x, y - size), QPointF(x + thin, y - thin), QPointF(x + size, y),
                QPointF(x + thin, y + thin), QPointF(x, y + size), QPointF(x - thin, y + thin),
                QPointF(x - size, y), QPointF(x - thin, y - thin),
            ]))

    @staticmethod
    def _paint_scene(p: QPainter, at, unit: float, t: float, alpha: float, palette: dict,
                     fonts) -> None:
        """The ground, the button on its pedestal, and the scientist, in line
        drawing: the theme's text colour for lines, its card colour inside."""
        ink = palette.get("text", "#e6edf3")
        paper = palette.get("surface", "#11161d")
        accent = palette.get("accent", "#58a6ff")
        line = max(1.4, fonts.px(2) * 0.9)
        p.setOpacity(alpha)

        p.setPen(QPen(QColor(palette.get("border", ink)), line))
        p.drawLine(at(-1.15, GROUND), at(1.15, GROUND))

        # The button: a pedestal and a cap that lights.
        shine = lit(t)
        cap_y = CAP_Y + cap_offset(t)
        p.setPen(QPen(QColor(ink), line))
        p.setBrush(QColor(palette.get("surface2", paper)))
        p.drawRoundedRect(QRectF(at(BUTTON_X - BUTTON_HALF, CAP_Y), at(BUTTON_X + BUTTON_HALF, GROUND)),
                          unit * 0.014, unit * 0.014)
        # A small star on its front: this button is for the star.
        badge = star_outline(0.035) + (BUTTON_X, (CAP_Y + GROUND) / 2 + 0.005)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(QPolygonF([at(x, y) for x, y in badge]))
        if shine > 0:
            halo_at = at(BUTTON_X, cap_y - CAP_RY / 2)
            halo = QRadialGradient(halo_at, unit * 0.14)
            halo.setColorAt(0.0, _with_alpha(accent, 0.45 * shine))
            halo.setColorAt(1.0, _with_alpha(accent, 0.0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(halo)
            p.drawEllipse(halo_at, unit * 0.14, unit * 0.14)
        p.setPen(QPen(QColor(ink), line))
        p.setBrush(_with_alpha(accent, 0.45 + 0.55 * shine))
        cap = QPainterPath()
        cap.moveTo(at(BUTTON_X - CAP_RX, CAP_Y))
        cap.arcTo(QRectF(at(BUTTON_X - CAP_RX, cap_y - CAP_RY), at(BUTTON_X + CAP_RX, cap_y + CAP_RY)),
                  180, -180)
        cap.closeSubpath()
        p.drawPath(cap)

        if t >= WALK_AT:
            StarCanvas._paint_scientist(p, at, unit, t, palette, line)
        p.setOpacity(1.0)

    @staticmethod
    def _paint_scientist(p: QPainter, at, unit: float, t: float, palette: dict,
                         line: float) -> None:
        """A small scientist in a lab coat, in line drawing: the theme's text
        colour for lines, its card colour inside, the accent for the glasses,
        the tie and a pen."""
        ink = QColor(palette.get("text", "#e6edf3"))
        paper = QColor(palette.get("surface", "#11161d"))
        accent = QColor(palette.get("accent", "#58a6ff"))
        quiet = QColor(palette.get("dim", palette.get("text", "#8b949e")))
        pose = pose_at(t)
        sx, sy = shoulder(pose)
        bsx, bsy = sx - 0.022, sy + 0.004
        pen = QPen(ink, line * 1.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        thin = QPen(ink, line * 0.75, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin)

        def seg(a, b) -> None:
            p.drawLine(at(*a), at(*b))

        def disc(c, r: float, fill) -> None:
            p.setBrush(fill)
            p.drawEllipse(at(*c), unit * r, unit * r)

        def arm(shoulder_at, hand) -> None:
            elbow = joint(shoulder_at, hand, UPPER, FORE, 1.0)
            p.setPen(pen)
            seg(shoulder_at, elbow)
            seg(elbow, hand)
            disc(hand, 0.011, paper)

        arm((bsx, bsy), pose.hands[0])              # behind the body
        # The legs and the shoes.
        for side, foot in zip((-0.012, 0.012), pose.feet):
            hip = (pose.x + side, pose.hip_y)
            knee = joint(hip, foot, THIGH, SHIN, -1.0)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            seg(hip, knee)
            seg(knee, foot)
            p.setBrush(ink)
            p.drawRoundedRect(QRectF(at(foot[0] - 0.012, foot[1] - 0.005),
                                     at(foot[0] + 0.03, foot[1] + FOOT_H)),
                              unit * 0.006, unit * 0.006)
        # The lab coat: collar and lapels, a tie, two buttons, a pocket with pens.
        hem = pose.hip_y + 0.06
        p.setPen(pen)
        p.setBrush(paper)
        p.drawPolygon(QPolygonF([at(sx - 0.046, sy + 0.004), at(sx + 0.044, sy),
                                 at(pose.x + 0.07, hem), at(pose.x - 0.07, hem)]))
        v = (sx + 0.014, sy + 0.052)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent)
        p.drawPolygon(QPolygonF([at(sx + 0.008, sy + 0.006), at(sx + 0.02, sy + 0.006),
                                 at(sx + 0.017, sy + 0.014), at(sx + 0.021, sy + 0.044),
                                 at(sx + 0.014, sy + 0.05), at(sx + 0.007, sy + 0.044),
                                 at(sx + 0.011, sy + 0.014)]))
        p.setPen(thin)
        p.setBrush(Qt.BrushStyle.NoBrush)
        seg((sx - 0.014, sy + 0.003), v)
        seg((sx + 0.036, sy + 0.001), v)
        front_edge = (pose.x + 0.02, hem)
        seg(v, front_edge)
        p.setBrush(ink)
        for f in (0.35, 0.68):
            bx = v[0] + (front_edge[0] - v[0]) * f + 0.01
            by = v[1] + (front_edge[1] - v[1]) * f
            p.drawEllipse(at(bx, by), unit * 0.0045, unit * 0.0045)
        p.setBrush(paper)
        p.drawRect(QRectF(at(sx - 0.038, sy + 0.04), at(sx - 0.014, sy + 0.062)))
        for px_, colour, top in ((sx - 0.031, accent, 0.026), (sx - 0.022, quiet, 0.03)):
            p.setPen(QPen(colour, line * 0.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            seg((px_, sy + 0.04), (px_ + 0.001, sy + top))
        # The head: neck, face, ear, hair, glasses, nose and a smile.
        p.setPen(pen)
        seg((sx + 0.004, sy), (sx + 0.006, sy - NECK))
        hx, hy = sx + 0.008, sy - NECK - HEAD_R
        disc((hx, hy), HEAD_R, paper)
        # The ear: a small arc at the back of the head.
        p.setPen(thin)
        ear = QPainterPath(at(hx - 0.03, hy - 0.006))
        ear.quadTo(at(hx - 0.052, hy + 0.004), at(hx - 0.032, hy + 0.018))
        p.drawPath(ear)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k, deg in enumerate((165, 188, 210, 233, 256, 279)):
            a = np.deg2rad(deg)
            base = (hx + HEAD_R * np.cos(a), hy + HEAD_R * np.sin(a))
            length = 0.026 + 0.01 * (k % 2)
            tip = (base[0] + length * np.cos(a - 0.35), base[1] + length * np.sin(a - 0.35))
            bend = (base[0] + 0.6 * length * np.cos(a + 0.25), base[1] + 0.6 * length * np.sin(a + 0.25))
            tuft = QPainterPath(at(*base))
            tuft.quadTo(at(*bend), at(*tip))
            p.drawPath(tuft)
        look = -0.008 if pose.looks_up else 0.0
        eye = (hx + 0.033, hy - 0.008 + look)
        p.setPen(Qt.PenStyle.NoPen)
        disc(eye, 0.0055, ink)
        p.setPen(QPen(accent, line * 0.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(at(*eye), unit * 0.018, unit * 0.018)
        seg((eye[0] - 0.018, eye[1]), (hx - 0.01, hy - 0.002))       # the glasses' arm
        p.setPen(pen)
        nose = QPainterPath(at(hx + 0.062, hy - 0.004))
        nose.quadTo(at(hx + 0.084, hy + 0.012), at(hx + 0.061, hy + 0.017))
        p.drawPath(nose)
        smile = QPainterPath(at(hx + 0.03, hy + 0.034))
        smile.quadTo(at(hx + 0.045, hy + 0.04 + 0.016 * pose.smile), at(hx + 0.058, hy + 0.03))
        p.drawPath(smile)
        arm((sx, sy), pose.hands[1])                # in front of the coat


#: What the dialog says, item by item (also read by the tests).
ITEMS = (
    ("How",
     "Click Star on GitHub below. On the project's page, sign in if GitHub asks "
     "you to, and click Star at the top right. That is all. To hear of every new "
     "version as well, choose Watch, then Custom, then Releases."),
    ("Why it helps",
     "Stars make BIDS Manager more visible, so more of the people who need it can "
     "find it. And the project stays in your stars, one click away, so following "
     "its updates is easier."),
    ("Your privacy",
     "BIDS Manager does not track you. It collects no usage data and sends nothing "
     "about you or your data anywhere, so we cannot tell who uses it. Stars are "
     "the only way we get an idea of how many people find it useful."),
)


class StarDialog(QDialog):
    """Why a star on GitHub helps, and how to give one, beside the animation."""

    def __init__(self, parent: Optional[QWidget] = None, *, asked: bool = False) -> None:
        super().__init__(parent)
        from .app_settings import AppSettings
        from .viz import fonts

        self.setObjectName("star-dialog")
        self.setWindowTitle("A star for BIDS Manager")
        self.setModal(True)
        self.resize(fonts.px(900), fonts.px(540))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(fonts.px(16), fonts.px(16), fonts.px(28), fonts.px(8))
        body.setSpacing(fonts.px(20))
        self.canvas = StarCanvas(self)
        body.addWidget(self.canvas, 10)

        text = QVBoxLayout()
        text.setSpacing(fonts.px(6))
        kicker = QLabel("If BIDS Manager helps you")
        kicker.setObjectName("whatsnew-kicker")
        title = QLabel("Give it a star on GitHub")
        title.setObjectName("whatsnew-title")
        lede = QLabel(
            "If BIDS Manager is helpful and valuable in your work, a star on GitHub is "
            "the easiest way to tell us. It takes a few seconds.")
        lede.setObjectName("whatsnew-lede")
        lede.setWordWrap(True)
        text.addStretch(1)
        for widget in (kicker, title, lede):
            text.addWidget(widget)
        text.addSpacing(fonts.px(10))
        for heading, words in ITEMS:
            head = QLabel(heading)
            head.setObjectName("whatsnew-item-title")
            line = QLabel(words)
            line.setObjectName("whatsnew-item")
            line.setWordWrap(True)
            text.addWidget(head)
            text.addWidget(line)
            text.addSpacing(fonts.px(6))
        text.addStretch(1)
        body.addLayout(text, 9)
        outer.addLayout(body, 1)

        footer = QFrame(self)
        footer.setObjectName("whatsnew-footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(fonts.px(16), fonts.px(10), fonts.px(16), fonts.px(10))
        self.never = QCheckBox("Don't ask me again", footer)
        self.never.setToolTip(
            "BIDS Manager asks at most once a day, until you have opened the project's "
            "page from here. The same switch is in Settings > Display. Home's Help us "
            "button brings this back at any time.")
        self.never.setChecked(not AppSettings.load().ask_for_star)
        self.never.toggled.connect(lambda on: AppSettings.remember_ask_for_star(not on))
        later = QPushButton("Not now" if asked else "Close", footer)
        later.setAutoDefault(False)
        later.clicked.connect(self.reject)
        star = QPushButton("Star on GitHub", footer)
        star.setDefault(True)
        star.setToolTip(REPO_URL)
        star.clicked.connect(self._go_to_github)
        row.addWidget(self.never)
        row.addStretch(1)
        row.addWidget(later)
        row.addWidget(star)
        outer.addWidget(footer)
        self._buttons = {"later": later, "star": star}
        star.setFocus()

    def _go_to_github(self) -> None:
        """Open the project's page; from now on, ask no more on our own."""
        schedule = _load()
        schedule.opened = True
        _save(schedule)
        QDesktopServices.openUrl(QUrl(REPO_URL))
        self.accept()


__all__ = ["COUNTS", "ITEMS", "JUMPS", "REPO_URL", "Pose", "StarCanvas", "StarDialog",
           "StarParticles", "StarSchedule", "dimness", "due", "form_points", "joint", "jump_at",
           "lit", "maybe_show", "pose_at", "show_dialog", "star_outline"]
