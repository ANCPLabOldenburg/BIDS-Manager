"""GPU volume-raycasting 3-D view for the NIfTI viewer.

Companion to :class:`bidsmgr.gui.widgets.NiftiViewerPane`. Renders the
current volume with a single-pass OpenGL ray-caster — the technique
MRIcroGL uses — with only the dependencies BIDS-Manager already ships
(PyQt6 + PyOpenGL). Cross-platform (macOS / Windows / Linux): the context
is a portable OpenGL 3.3 core profile and every GL call goes through
PyOpenGL, with no platform-specific paths.

Effects (``uEffect``), matching MRIcroGL's Render menu — approximations of
its ``Default`` / ``Matte`` / ``Glass`` / ``MIP`` / ``Edges`` /
``OpacityPeeling`` / ``Shell`` / ``Tomography`` shaders using on-the-fly
gradients:
    0 Default · 1 Matte · 2 Glass · 3 X-ray · 4 MIP · 5 Edges ·
    6 Opacity peeling · 7 Opacity peeling 2 · 8 Shell · 9 Topography

Lighting: matcap materials for the surface effects (Default/Topography/
Edges) plus Phong ambient/diffuse/specular/shininess with a positionable
light for the rest. An oblique clip plane (azimuth/elevation/depth/
thickness) slices the volume; for the surface effects the exposed face is
drawn as a smooth intensity cross-section *with depth* (dark structures
such as the ventricles show volume, not a flat cut), while the transparent
effects (Glass/X-ray/MIP/Edges) just clip.

An orientation cube (S/I·A/P·L/R) is drawn in the corner (toggle with "O").

GPU gate: :func:`gpu_available` reports whether an OpenGL 3.3 core context
backed by real hardware exists; the pane hides all 3-D options when not.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QSurfaceFormat, QImage, QPainter, QColor, QFont
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


def _gl_format() -> QSurfaceFormat:
    """A portable OpenGL 3.3 core surface format (does not touch the default)."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    return fmt


def request_gl_format() -> QSurfaceFormat:
    """Register the 3.3 core format as the app default (call once, pre-app).

    Widgets use :func:`_gl_format` for their own context so the default is not
    re-set after the shared context exists (which Qt warns about).
    """
    fmt = _gl_format()
    QSurfaceFormat.setDefaultFormat(fmt)
    return fmt


_GPU_CACHE: Optional[bool] = None


def gpu_available() -> bool:
    """True when an OpenGL 3.3-core context on real hardware is reachable.

    Cross-platform probe (macOS/Windows/Linux): create a throwaway offscreen
    context, confirm it reports >= 3.3, and reject known software rasterisers
    (llvmpipe / softpipe / Microsoft GDI). Cached after the first call.
    Requires a live ``QApplication`` — returns ``False`` if none exists yet.
    """
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    _GPU_CACHE = False
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext
        if QApplication.instance() is None:
            return False
        fmt = _gl_format()
        surf = QOffscreenSurface()
        surf.setFormat(fmt)
        surf.create()
        if not surf.isValid():
            return False
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if ctx.create() and ctx.makeCurrent(surf):
            f = ctx.format()
            from OpenGL import GL
            renderer = (GL.glGetString(GL.GL_RENDERER) or b"").decode(errors="ignore")
            ok = (f.majorVersion(), f.minorVersion()) >= (3, 3)
            soft = any(s in renderer.lower()
                       for s in ("llvmpipe", "softpipe", "software", "gdi generic"))
            ctx.doneCurrent()
            _GPU_CACHE = bool(ok and not soft)
        surf.destroy()
    except Exception as exc:  # noqa: BLE001
        log.info("GPU probe failed (%s); hiding 3-D options.", exc)
        _GPU_CACHE = False
    return _GPU_CACHE


# Effect dropdown labels (display order). "Standard" is the default. Some
# entries are *presets* of an existing shader effect (Jelly / Skull are named
# parameter presets of Opacity peeling). The display label is decoupled from
# the shader ``uEffect`` index via :data:`EFFECT_FX`, so labels/order can change
# without touching the shader.
EFFECTS = ["Standard", "Matte", "Juicy shiny", "Juicy shiny 2", "Realistic",
           "Glass", "X-ray", "Jelly", "Skull", "MIP", "Edges",
           "Opacity peeling", "Opacity peeling 2", "Shell", "Topography"]

# Label -> shader effect index (uEffect). "Standard" = flat Phong (shader fx 1),
# "Matte" = waxy matcap (shader fx 0). "Juicy shiny" is a Standard preset;
# Jelly/Skull reuse Opacity peeling (fx 6).
EFFECT_FX: dict[str, int] = {
    "Standard": 1, "Matte": 0, "Juicy shiny": 1, "Juicy shiny 2": 1,
    "Realistic": 10,
    "Glass": 2, "X-ray": 3, "MIP": 4, "Edges": 5,
    "Opacity peeling": 6, "Opacity peeling 2": 7, "Shell": 8, "Topography": 9,
    "Jelly": 6, "Skull": 6,
}

# Which control keys each effect uses (others are greyed out in the panel).
# "overlay"/"overlaydepth" (the cut-face intensity slice) only appear for the
# opaque surface effects where a cross-section reads sensibly.
_COMMON = {"lo", "hi", "quality"}
_OVERLAY = {"overlay", "overlaydepth"}
_MATCAP_PARAMS = _COMMON | _OVERLAY | {"density", "light", "brighten", "surface",
                                       "ambient", "diffuse", "specular",
                                       "shininess", "lightaz", "lightel"}
_PHONG_PARAMS = _COMMON | _OVERLAY | {"density", "ambient", "diffuse", "specular",
                                      "shininess", "lightaz", "lightel"}
_PEEL_PARAMS = _COMMON | {"density", "ambient", "diffuse", "specular", "shininess",
                          "peel", "tlow", "thigh", "lightaz", "lightel"}
EFFECT_PARAMS: dict[str, set] = {
    "Standard":          _PHONG_PARAMS,     # flat Phong (was "Matte")
    "Matte":             _MATCAP_PARAMS,    # waxy matcap (was "Default")
    "Juicy shiny":       _PHONG_PARAMS,     # Standard preset (wet, glossy tissue)
    "Juicy shiny 2":     _PHONG_PARAMS,     # sharper, harder highlight
    "Realistic":         _PHONG_PARAMS,     # living-tissue colour ramp
    "Glass":             _COMMON | {"specular", "shininess", "edgethresh",
                                    "boundthresh", "edgemix", "colortemp",
                                    "lightaz", "lightel"},
    "X-ray":             _COMMON | {"density"},
    "MIP":               _COMMON,
    "Edges":             _COMMON | {"density", "light", "brighten", "surface",
                                    "boundthresh", "edgethresh", "edgemix",
                                    "lightaz", "lightel"},
    "Opacity peeling":   _PEEL_PARAMS,
    "Opacity peeling 2": _PEEL_PARAMS,
    "Shell":             _COMMON | {"boundthresh", "edgethresh", "edgemix",
                                    "colortemp", "specular", "lightaz", "lightel"},
    "Topography":        _COMMON | _OVERLAY | {"density", "light", "brighten",
                                               "surface", "gradientmix",
                                               "intensitymix", "hardness",
                                               "lightaz", "lightel"},
    "Jelly":             _PEEL_PARAMS,
    "Skull":             _PEEL_PARAMS,
}
# Effects whose cut-face intensity slice is ON by default.
SLICE_DEFAULT_ON = {"Standard", "Matte", "Juicy shiny", "Juicy shiny 2",
                    "Realistic", "Topography"}

# Named parameter presets (control key -> value) applied when a preset effect
# is selected. Derived from the reference MRIcroGL-style looks.
#
# Presets deliberately leave out two things, so each has ONE source of truth:
#   * "quality" — a performance knob, not a look; every effect starts at
#     DEFAULTS["quality"] (50%) and the user raises it when they want.
#   * the clip plane — global state driven by its own controls / slicer
#     shortcuts; baking it in would hijack the slicer on every effect change.
EFFECT_PRESET: dict[str, dict] = {
    # Wet, glossy surface with a bright cut-face slice — a Standard (flat
    # Phong) preset.
    "Juicy shiny": dict(lo=36, hi=303, density=150, ambient=94, diffuse=50,
                        specular=50, shininess=20, lightaz=0, lightel=30,
                        overlay=True, overlaydepth=68),
    # Same window, but a much tighter/harder specular — a sharper wet sheen.
    "Juicy shiny 2": dict(lo=36, hi=303, density=150, ambient=94, diffuse=23,
                          specular=96, shininess=100, lightaz=0, lightel=30,
                          overlay=True, overlaydepth=68),
    # "Juicy shiny 2" lighting on the living-tissue shader. Ambient is pulled
    # down and diffuse up versus its parent: the flat, heavily-ambient look
    # that reads fine in grey turns a coloured surface into a flat pink card,
    # so the form has to come from directional light instead.
    "Realistic": dict(lo=36, hi=303, density=150, ambient=52, diffuse=78,
                      specular=62, shininess=80, lightaz=0, lightel=30,
                      overlay=True, overlaydepth=68),
    # Translucent, glassy tissue — the brain shows through the skin.
    "Jelly": dict(lo=110, hi=400, density=7, ambient=115, diffuse=65,
                  specular=35, shininess=45, peel=1, tlow=13, thigh=82,
                  lightaz=0, lightel=25),
    # Peel the skin to reveal the deeper (facial / orbital / bony) anatomy.
    "Skull": dict(lo=36, hi=303, density=150, ambient=95, diffuse=50,
                  specular=50, shininess=20, peel=1, tlow=19, thigh=80,
                  lightaz=0, lightel=30),
}

# Matcap "lighting" materials: (ambient, key, fill, spec_power, spec_int, tint).
_LIGHTINGS: dict[str, tuple] = {
    "Shiny White": (0.32, 0.90, 0.24, 55.0, 1.55, (1.00, 0.98, 0.95)),
    "Clay":        (0.38, 0.64, 0.28, 12.0, 0.16, (0.86, 0.78, 0.70)),
    "Bone":        (0.40, 0.66, 0.26, 28.0, 0.35, (0.94, 0.91, 0.84)),
    "Titanium":    (0.24, 0.82, 0.22, 96.0, 0.95, (0.72, 0.75, 0.80)),
    "Gold":        (0.30, 0.74, 0.22, 44.0, 0.85, (0.96, 0.80, 0.36)),
    "Blue":        (0.30, 0.68, 0.28, 34.0, 0.60, (0.56, 0.69, 0.96)),
}
LIGHTINGS = list(_LIGHTINGS)

# Slider range for Quality (ray-march steps). QUALITY_DEFAULT is half of the
# maximum: every effect starts at 50% so the first render is responsive on
# modest GPUs, and the user can push the slider to full when they want it.
QUALITY_MIN, QUALITY_MAX = 64, 1024
QUALITY_DEFAULT = QUALITY_MAX // 2

# Default control values ("Reset parameters").
DEFAULTS = dict(
    effect=0, light=0, lo=100, hi=420, density=100, brighten=150, surface=70,
    ambient=60, diffuse=55, specular=30, shininess=40,
    boundthresh=30, edgethresh=12, edgemix=65, colortemp=50,
    gradientmix=60, intensitymix=35, hardness=50,
    peel=1, tlow=25, thigh=85, lightaz=0, lightel=0, overlay=True, overlaydepth=28,
    quality=QUALITY_DEFAULT, clip=False, clipaz=0, clipel=0, depth=500,
    thick=1000, cube=True,
)


# --------------------------------------------------------------------------
# Volume normalisation + geometry helpers
# --------------------------------------------------------------------------

def normalize_to_u8(vol: np.ndarray) -> np.ndarray:
    """Window a float volume to uint8 on a robust 0.5–99.5 percentile range."""
    v = np.ascontiguousarray(vol, dtype=np.float32)
    lo, hi = np.percentile(v, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi <= lo:
            hi = lo + 1.0
    return (np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def clip_normal_from(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Unit clip-plane normal from azimuth/elevation (RAS texcoord space)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    n = _normalize(np.array([ce * np.sin(az), ce * np.cos(az), np.sin(el)], np.float32))
    return float(n[0]), float(n[1]), float(n[2])


def light_dir_view(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Light direction in view space (x right, y up, z toward camera)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    d = _normalize(np.array([ce * np.sin(az), np.sin(el), ce * np.cos(az)], np.float32))
    return float(d[0]), float(d[1]), float(d[2])


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _ortho(r: float, t: float, n: float, f: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 / r
    m[1, 1] = 1.0 / t
    m[2, 2] = -2.0 / (f - n)
    m[2, 3] = -(f + n) / (f - n)
    return m


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(center - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _make_matcap(name: str, size: int = 256) -> np.ndarray:
    """Procedural studio-lit sphere → an RGB matcap (``size×size×3`` uint8)."""
    amb, key_i, fill_i, spow, spec_i, tint = _LIGHTINGS[name]
    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    u, v = np.meshgrid(ax, ax)
    r2 = u * u + v * v
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    n = np.stack([u, v, z], axis=-1)
    key = _normalize(np.array([0.35, 0.55, 0.75], np.float32))
    fill = _normalize(np.array([-0.6, 0.10, 0.55], np.float32))
    view = np.array([0.0, 0.0, 1.0], np.float32)
    half = _normalize(key + view)
    ndl_key = np.clip((n * key).sum(-1), 0.0, 1.0)
    ndl_fill = np.clip((n * fill).sum(-1), 0.0, 1.0)
    nh = np.clip((n * half).sum(-1), 0.0, 1.0)
    spec = nh ** spow + 0.25 * (nh ** (spow * 0.25))
    wrap = np.clip((ndl_key + 0.35) / 1.35, 0.0, 1.0)
    lum = amb + key_i * wrap + fill_i * ndl_fill
    col = lum[..., None] * np.array(tint, np.float32)
    col = col + spec[..., None] * (spec_i * np.array([1.0, 1.0, 1.0], np.float32))
    col = np.clip(col, 0.0, 1.0)
    col[~inside] = 0.0
    return (col * 255.0).astype(np.uint8)


def _make_cube_atlas(size: int = 96, flip=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Render the 6 orientation letters into a 3×2 RGBA atlas (numpy).

    ``flip`` (per L/R, A/P, S/I axis; -1 flips) swaps the letter pair on that
    axis so a mirrored render shows the correct label upright — the cube is
    never geometrically reflected (that would mirror the glyphs).
    """
    letters = [["R", "A", "S"], ["L", "P", "I"]]
    for col, f in enumerate(flip):
        if f < 0:
            letters[0][col], letters[1][col] = letters[1][col], letters[0][col]
    img = QImage(size * 3, size * 2, QImage.Format.Format_RGBA8888)
    img.fill(QColor(38, 44, 54))
    p = QPainter(img)
    f = QFont()
    f.setPixelSize(int(size * 0.6))
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(235, 240, 248))
    for row in range(2):
        for col in range(3):
            rect = img.rect().adjusted(col * size, row * size, 0, 0)
            rect.setWidth(size)
            rect.setHeight(size)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, letters[row][col])
    p.end()
    img = img.mirrored(False, True)  # flip for GL bottom-up texture coords
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.width(), 4).copy()
    return arr


def _cube_geometry() -> np.ndarray:
    """Interleaved [pos3, normal3, uv2] for a labelled orientation cube.

    Atlas cells (3×2): R A S / L P I. Each face's in-plane "up" axis is chosen
    so the letter reads upright when that face points at the viewer — S
    (superior, +Z) for the four side faces, A (anterior, +Y) for top/bottom.
    Depth testing (not culling) resolves visibility, so winding is free.
    """
    cells = {"R": (0, 0), "A": (1, 0), "S": (2, 0), "L": (0, 1), "P": (1, 1), "I": (2, 1)}
    faces = [
        ("R", (1, 0, 0), (0, 0, 1)),
        ("L", (-1, 0, 0), (0, 0, 1)),
        ("A", (0, 1, 0), (0, 0, 1)),
        ("P", (0, -1, 0), (0, 0, 1)),
        ("S", (0, 0, 1), (0, 1, 0)),
        ("I", (0, 0, -1), (0, 1, 0)),
    ]
    verts = []
    for letter, nrm, up in faces:
        n = np.array(nrm, np.float32)
        upv = np.array(up, np.float32)
        right = np.cross(upv, n)
        col, vrow = cells[letter]
        u0, v0 = col / 3.0, (1 - vrow) * 0.5   # atlas flipped: row 0 -> high v

        def corner(iu, iv):
            pos = n + (2 * iu - 1) * right + (2 * iv - 1) * upv
            return [*pos, *nrm, u0 + iu / 3.0, v0 + iv * 0.5]

        c = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]
        for a, b, d in ((0, 1, 2), (0, 2, 3)):
            verts += [c[a], c[b], c[d]]
    return np.array(verts, np.float32)


# --------------------------------------------------------------------------
# Shaders
# --------------------------------------------------------------------------

_VERT = """
#version 330 core
out vec2 vNdc;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vNdc = pos * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 0.0, 1.0);
}
"""

_FRAG = """
#version 330 core
in  vec2 vNdc;
out vec4 FragColor;

uniform sampler3D uVol;
uniform int   uIsRGB;      // 1 when uVol carries colour (e.g. colour-FA)
uniform sampler2D uMatcap;
uniform mat4  uInvViewProj;
uniform mat3  uNormalMatrix;
uniform vec3  uBoxHalf;
uniform vec3  uTexSize;
uniform int   uEffect;
uniform float uThreshLo, uThreshHi, uDensity;
uniform float uBrighten, uSurface;
uniform float uAmbient, uDiffuse, uSpecular, uShininess;
uniform float uBoundThresh, uEdgeThresh, uEdgeMix, uColorTemp;
uniform float uGradientMix, uIntensityMix, uHardness;
uniform int   uPeel; uniform float uTlow, uThigh;
uniform vec3  uLightDir;
uniform int   uSteps;
uniform vec3  uBg;
uniform int   uClipActive;
uniform vec3  uClipNormal;
uniform float uClipDepth, uClipThick;
uniform int   uSliceOverlay;   // draw the cut face as an intensity slice
uniform float uSliceDepth;     // 0..1 -> how deep the slice integrates

bool intersectBox(vec3 ro, vec3 rd, out float tN, out float tF) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (-uBoxHalf - ro) * inv, t1 = (uBoxHalf - ro) * inv;
    vec3 a = min(t0, t1), b = max(t0, t1);
    tN = max(max(a.x, a.y), a.z);
    tF = min(min(b.x, b.y), b.z);
    return tF >= max(tN, 0.0);
}
// Scalar density at p. For an RGB volume (colour-FA and friends) the three
// channels are a direction scaled by anisotropy, so the VECTOR LENGTH is the
// direction-independent magnitude — exactly the FA for a colour-FA map. A
// luminance weighting would be wrong here: it would make green (A-P) fibres
// several times denser than blue (S-I) ones purely because of their hue.
float samp(vec3 p) {
    vec4 t = texture(uVol, p);
    return uIsRGB == 1 ? clamp(length(t.rgb), 0.0, 1.0) : t.r;
}
// Voxel hue, normalised to full brightness so lighting (not the raw channel
// magnitude) sets how light or dark the surface reads. White for scalar data.
vec3 voxelTint(vec3 p) {
    if (uIsRGB == 0) return vec3(1.0);
    vec3 c = texture(uVol, p).rgb;
    float m = max(max(c.r, c.g), c.b);
    return m > 0.0031 ? c / m : vec3(1.0);
}
bool clipped(vec3 p) {
    if (uClipActive == 0) return false;
    float sd = dot(uClipNormal, p - vec3(0.5));
    return sd > uClipDepth && sd < uClipDepth + uClipThick;
}
vec3 grad(vec3 p) {
    vec3 e = 1.0 / uTexSize;
    return vec3(samp(p+vec3(e.x,0,0))-samp(p-vec3(e.x,0,0)),
                samp(p+vec3(0,e.y,0))-samp(p-vec3(0,e.y,0)),
                samp(p+vec3(0,0,e.z))-samp(p-vec3(0,0,e.z)));
}
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }
// Living-brain tissue colour by intensity. A fresh brain is not grey: pia and
// the vessels over it are dark red, cortical grey matter is a dull pinkish
// mauve, and white matter / fatty tissue is a pale warm cream. Ramping T1
// intensity through those three anchors reads as real tissue instead of
// tinted stone. Deliberately desaturated at the top end — pushing white
// matter toward saturated pink is what makes fake-looking "meat" renders.
vec3 brainTissue(float d) {
    // Sampled from a fixed coronal specimen. Real brain is far LIGHTER and
    // far LESS saturated than intuition suggests: white matter is close to
    // ivory, cortex only a muted tan-grey, and only the vessels are properly
    // red. Saturated pink everywhere is the classic "raw meat" failure.
    //
    // The floor is deliberately a dusky rose rather than anything near black:
    // a cut brain has no black regions (sulci read as thin red lines, CSF as
    // gaps), so letting the ramp fall to black paints holes in the tissue.
    const vec3 DEEP   = vec3(0.55, 0.33, 0.30);   // sulcal shadow / vessel bed
    const vec3 CORTEX = vec3(0.82, 0.69, 0.63);   // grey matter, muted tan
    const vec3 WHITE  = vec3(0.97, 0.94, 0.89);   // white matter, ivory
    // The white transition is centred on the measured grey/white boundary of
    // a percentile-normalised T1 (grey matter lands near 0.40, white matter
    // near 0.58 here). Starting it higher, as a first pass did, left white
    // matter only a third of the way to ivory and the whole slice read as one
    // flat tan — the grey/white contrast is most of what makes it legible.
    float x = clamp(d, 0.0, 1.0);
    vec3 c = mix(DEEP, CORTEX, smoothstep(0.02, 0.30, x));
    return mix(c, WHITE, smoothstep(0.44, 0.62, x));
}
vec3 tempTint(float t){
    vec3 c = vec3(1.0);
    if (t < 0.5) { c.b = 1.0+(0.5-t); c.r = 1.0-(0.5-t)*0.6; }
    else         { c.r = 1.0+(t-0.5); c.b = 1.0-(t-0.5)*0.6; }
    return clamp(c, 0.0, 1.4);
}

void main() {
    vec4 pn = uInvViewProj * vec4(vNdc, -1.0, 1.0);
    vec4 pf = uInvViewProj * vec4(vNdc,  1.0, 1.0);
    vec3 ro = pn.xyz / pn.w;
    vec3 rd = normalize(pf.xyz / pf.w - ro);
    float tN, tF;
    if (!intersectBox(ro, rd, tN, tF)) { FragColor = vec4(uBg,1.0); return; }
    tN = max(tN, 0.0);
    vec3 boxSize = 2.0 * uBoxHalf;
    float dt = length(boxSize) / float(uSteps);
    float refStep = length(boxSize) / 512.0;
    float t0 = tN + hash(gl_FragCoord.xy) * dt;
    vec3 duvw = rd * dt / boxSize;

    float e0 = min(uThreshLo, uThreshHi);
    float e1 = max(uThreshLo, uThreshHi);
    if (e1 <= e0) e1 = e0 + 1e-3;

    // ---- MIP ----
    if (uEffect == 4) {
        float mx = 0.0, t = t0; vec3 tint = vec3(1.0);
        for (int i=0;i<4096;++i){ if(t>tF)break; vec3 p=(ro+rd*t+uBoxHalf)/boxSize;
            // Carry the hue of the brightest voxel so a colour-FA MIP keeps
            // its direction colouring instead of collapsing to grey.
            if(!clipped(p)){ float s=samp(p); if(s>mx){ mx=s; tint=voxelTint(p);} } t+=dt; }
        float w = clamp((mx-e0)/(e1-e0),0.0,1.0);
        FragColor = vec4(mix(uBg, tint, w), 1.0); return;
    }
    // ---- X-ray ----
    if (uEffect == 3) {
        float sum=0.0, t=t0; vec3 csum = vec3(0.0);
        for (int i=0;i<4096;++i){ if(t>tF)break; vec3 p=(ro+rd*t+uBoxHalf)/boxSize;
            if(!clipped(p)){ float d=samp(p);
                if(d>e0){ float w=(d-e0)*uDensity; sum+=w; csum+=voxelTint(p)*w; } }
            t+=dt; }
        float a = 1.0 - exp(-sum*dt*6.0);
        // Density-weighted mean hue along the ray (white for scalar volumes).
        vec3 tint = sum > 1e-6 ? csum / sum : vec3(1.0);
        FragColor = vec4(mix(uBg, tint, clamp(a,0.0,1.0)), 1.0); return;
    }

    vec3 viewDir = vec3(0.0, 0.0, 1.0);

    // ---- Opacity peeling (Rezk-Salama & Kolb): render the (peel+1)-th layer.
    //      Peeling (6) resets the accumulator on each layer; Peeling 2 (7)
    //      keeps prior layers faintly (translucent nested peel).
    if (uEffect==6 || uEffect==7) {
        vec4 acc = vec4(0.0); float pNum = 0.0; float t = t0;
        for (int i=0;i<4096;++i){
            if (t>tF) break;
            vec3 p = (ro+rd*t+uBoxHalf)/boxSize;
            if (clipped(p)) { t+=dt; continue; }
            float d = samp(p);
            float a = smoothstep(e0,e1,d) * uDensity;
            a = 1.0 - pow(1.0-clamp(a,0.0,1.0), dt/refStep);
            if (a > 0.01) {
                vec3 nv = normalize(uNormalMatrix * normalize(-grad(p)+1e-6));
                float ndl = max(dot(nv, uLightDir), 0.0);
                float sp = pow(max(dot(reflect(-uLightDir,nv),viewDir),0.0), max(uShininess,1.0))*uSpecular;
                vec3 base = vec3(pow(d,0.8));
                vec3 lit = (base*(uAmbient + uDiffuse*ndl) + vec3(sp)) * voxelTint(p);
                acc.rgb += (1.0-acc.a)*lit*a; acc.a += (1.0-acc.a)*a;
            }
            if (acc.a > uThigh && a < uTlow) {            // filled a layer, then exited it
                pNum += 1.0;
                if (pNum > float(uPeel)) break;
                if (uEffect==6) acc = vec4(0.0);          // hard peel
                else { acc.rgb *= 0.30; acc.a *= 0.30; }  // soft (translucent) peel
            }
            t += dt;
        }
        FragColor = vec4(acc.rgb + (1.0-acc.a)*uBg, 1.0); return;
    }

    bool translucent = (uEffect==2 || uEffect==5 || uEffect==8);   // Glass/Edges/Shell
    vec3 edgeCol = tempTint(uColorTemp);

    vec4 acc = vec4(0.0);
    float t = t0;
    bool prevClip = false;
    for (int i=0;i<4096;++i){
        if (t>tF || (acc.a>0.985 && !translucent)) break;
        vec3 p = (ro+rd*t+uBoxHalf)/boxSize;
        if (clipped(p)) { prevClip = true; t += dt; continue; }
        float d = samp(p);

        // Cut face -> hybrid cross-section. SOLID tissue at the plane is drawn
        // as a clean flat intensity slice; low-intensity CSF / air is treated
        // as EMPTY (transparent) so the lit 3-D render behind shows through —
        // a ventricle becomes a real, shaded recess (that is the "depth"). The
        // overlay-depth slider raises the emptiness threshold; smoothstep keeps
        // the boundary smooth so there are no black-dot speckles. We do NOT
        // break: the loop continues and the normal surface shading below fills
        // the transparent parts. Scoped entirely to this block.
        if (uSliceOverlay==1 && uClipActive==1 && prevClip) {
            prevClip = false;
            // De-jitter: snap to the exact clip-plane crossing for a clean value.
            vec3 q = p;
            float denom = dot(uClipNormal, duvw);
            if (abs(denom) > 1e-6) {
                float sdp = dot(uClipNormal, p - vec3(0.5));
                float tgt = (abs(sdp - uClipDepth) <= abs(sdp - (uClipDepth+uClipThick)))
                            ? uClipDepth : (uClipDepth + uClipThick);
                q = clamp(p + duvw * ((tgt - sdp) / denom), vec3(0.0), vec3(1.0));
            }
            float sd = samp(q);
            float thr = mix(0.05, 0.42, uSliceDepth);
            float solid = smoothstep(thr, thr + 0.12, sd);
            if (solid > 0.003) {
                // The cut face is a windowed intensity slice — greyscale for
                // every effect except Realistic, where a grey cross-section
                // beside coloured tissue would break the illusion, so it runs
                // through the same tissue ramp as the surface.
                float sv = pow(clamp(sd,0.0,1.0), 0.8);
                vec3 face = (uEffect==10) ? brainTissue(sv) * voxelTint(q)
                                          : vec3(sv);
                acc.rgb += (1.0-acc.a) * face * solid;
                acc.a   += (1.0-acc.a) * solid;
            }
            t += dt;
            continue;   // let the lit 3-D volume fill the empty (cavity) parts
        }
        prevClip = false;

        vec3 g = grad(p);
        float gm = length(g);
        vec3 nv = normalize(uNormalMatrix * normalize(-g + 1e-6));
        float op = smoothstep(e0, e1, d) * uDensity;
        float depth01 = clamp((t-tN)/max(tF-tN,1e-3), 0.0, 1.0);
        float ndl = max(dot(nv, uLightDir), 0.0);
        float sp = pow(max(dot(reflect(-uLightDir, nv), viewDir), 0.0), max(uShininess,1.0));

        vec3 lit; float a = op;
        if (uEffect==0) {                                  // fx0 = "Matte" label (waxy matcap)
            vec3 mc = texture(uMatcap, nv.xy*0.5+0.5).rgb;
            vec3 surf = mix(vec3(0.74), vec3(pow(d,0.72)), uSurface);
            // Waxy matcap base lit with the SAME Phong model as Matte
            // (ambient + diffuse*N·L, plus specular/shininess), so the lighting
            // controls behave identically between Default and Matte.
            vec3 base = mc * surf * uBrighten;
            lit = base * (uAmbient + uDiffuse * ndl) + vec3(sp * uSpecular);
            lit *= 1.0 - 0.18 * depth01;
            a = op * 0.92;
        } else if (uEffect==9) {                           // Topography
            vec3 mc = texture(uMatcap, nv.xy*0.5+0.5).rgb;
            vec3 surf = mix(vec3(0.74), vec3(pow(d,0.72)), uSurface);
            lit = mc * surf * uBrighten;
            vec3 iShade = vec3(pow(d, mix(1.6, 0.5, uHardness))) * uBrighten;
            lit = mix(lit, iShade, uIntensityMix);
            lit += vec3(1.0) * sp * 0.4;
            op *= mix(1.0, clamp(gm*4.0, 0.0, 1.0), uGradientMix);
            a = op;
            lit *= 1.0 - 0.18 * depth01;
        } else if (uEffect==5) {                           // Edges: translucent + contours
            vec3 mc = texture(uMatcap, nv.xy*0.5+0.5).rgb;
            float edge = smoothstep(uEdgeThresh, 1.0, gm);
            float bound = smoothstep(uBoundThresh, 1.0, gm);
            vec3 surf = mix(vec3(0.6), vec3(pow(d,0.8)), uSurface);
            lit = mc * surf * uBrighten * (0.35 + 0.9*edge);
            a = op * mix(0.02, 0.9, mix(bound, edge, uEdgeMix));
        } else if (uEffect==2 || uEffect==8) {             // Glass / Shell
            float edge = smoothstep(uEdgeThresh, 1.0, gm);
            float bnd = (uEffect==2) ? ((gm>uBoundThresh)?pow(1.0-abs(nv.z),4.0):0.0)
                                     : smoothstep(uBoundThresh, 1.0, gm);
            float e = mix(bnd, edge, uEdgeMix);
            lit = edgeCol * (e * uBrighten + sp * uSpecular);
            a = e * (uEffect==2 ? 0.35 : 0.7);
        } else if (uEffect==10) {                          // Realistic (living tissue)
            // Same Phong response as Standard, but the base is tissue-coloured
            // and gains two touches that sell "alive":
            //  * a rim of subsurface red where the surface turns away from the
            //    viewer, mimicking light scattering through thin tissue;
            //  * a WHITE specular, because a wet surface reflects the light's
            //    colour, not the tissue's — tinting it pink looks like plastic.
            vec3 base = brainTissue(pow(d, 0.85));
            // Subsurface warmth only at grazing angles, and gently: at the
            // 0.45 it started out it tinted the whole surface salmon.
            float rim = pow(1.0 - abs(nv.z), 4.0);
            base = mix(base, base * vec3(1.14, 0.80, 0.75), 0.20 * rim);
            lit = base * (uAmbient + uDiffuse*ndl) + vec3(1.0, 0.97, 0.94)*(sp*uSpecular);
            lit *= 1.0 - 0.18 * depth01;
            a = op;
        } else {                                           // fx1 = "Standard" label (flat Phong)
            vec3 base = vec3(mix(0.5, pow(d,0.8), 0.7));
            lit = base * (uAmbient + uDiffuse*ndl) + vec3(sp*uSpecular);
            lit *= 1.0 - 0.30 * depth01;
            a = op;
        }
        lit *= voxelTint(p);          // white for scalar data, hue for colour-FA
        if (a > 0.0008) {
            a = 1.0 - pow(1.0 - clamp(a,0.0,1.0), dt/refStep);
            acc.rgb += (1.0-acc.a) * lit * a;
            acc.a   += (1.0-acc.a) * a;
        }
        t += dt;
    }
    FragColor = vec4(acc.rgb + (1.0-acc.a)*uBg, 1.0);
}
"""

_CUBE_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uProj;
uniform mat3 uRot;
out vec3 vN; out vec2 vUV;
void main(){
    vec3 p = uRot * aPos;
    vN = uRot * aNormal;
    vUV = aUV;
    gl_Position = uProj * vec4(p * 0.62, 1.0);
}
"""

_CUBE_FRAG = """
#version 330 core
in vec3 vN; in vec2 vUV;
uniform sampler2D uAtlas;
out vec4 FragColor;
void main(){
    float sh = 0.55 + 0.45 * clamp(vN.z, 0.0, 1.0);
    vec4 tx = texture(uAtlas, vUV);
    vec3 face = vec3(0.15, 0.18, 0.24) * sh;
    vec3 col = mix(face, vec3(0.96), tx.r);   // white letters over shaded face
    FragColor = vec4(col, 1.0);
}
"""


# --------------------------------------------------------------------------
# GL widget (bare canvas)
# --------------------------------------------------------------------------

class RaycastGLWidget(QOpenGLWidget):
    """QOpenGLWidget hosting the single-pass volume raycaster (no controls)."""

    # Emitted when the clip plane is changed from *inside* the widget (keyboard
    # shortcuts / Shift-drag / Shift-scroll), so the controls panel can mirror
    # its sliders. Not emitted for changes driven by the controls themselves.
    clip_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFormat(_gl_format())
        self._prog = 0
        self._vao = 0
        self._tex = 0
        self._matcap_tex = 0
        self._cube_prog = 0
        self._cube_vao = 0
        self._cube_vbo = 0
        self._cube_tex = 0
        self._cube_nverts = 0
        self._tex_dims = (1, 1, 1)
        self._box_half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._gl_ok = False
        self._gl_error = ""
        self._volume: Optional[tuple[np.ndarray, tuple[float, float, float]]] = None
        self._matcaps: dict[str, np.ndarray] = {}
        self._matcap_dirty = True
        self._show_cube = True
        # Display flip along the canonical RAS axes (L/R, A/P, S/I). Driven by
        # the pane's RAS (native orientation) and Radiological toggles so the
        # 3-D render mirrors in lock-step with the 2-D slices. (1, 1, 1) is the
        # neurological RAS default. The orientation cube's letters are swapped
        # to match rather than mirrored, so they always read upright.
        self._flip = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._cube_flip = (1.0, 1.0, 1.0)   # last flip baked into the atlas

        self._az, self._el, self._dist = -0.6, 0.3, 1.9
        self._target = np.zeros(3, np.float32)
        self._last: Optional[QPoint] = None
        self._drag = "rotate"

        self.effect = EFFECT_FX[EFFECTS[DEFAULTS["effect"]]]   # shader fx of default
        self.thresh_lo = DEFAULTS["lo"] / 1000.0
        self.thresh_hi = DEFAULTS["hi"] / 1000.0
        self.density = DEFAULTS["density"] / 100.0
        self.brighten = DEFAULTS["brighten"] / 100.0
        self.surface = DEFAULTS["surface"] / 100.0
        self.ambient = DEFAULTS["ambient"] / 100.0
        self.diffuse = DEFAULTS["diffuse"] / 100.0
        self.specular = DEFAULTS["specular"] / 100.0
        self.shininess = float(DEFAULTS["shininess"])
        self.bound_thresh = DEFAULTS["boundthresh"] / 100.0
        self.edge_thresh = DEFAULTS["edgethresh"] / 100.0
        self.edge_mix = DEFAULTS["edgemix"] / 100.0
        self.color_temp = DEFAULTS["colortemp"] / 100.0
        self.gradient_mix = DEFAULTS["gradientmix"] / 100.0
        self.intensity_mix = DEFAULTS["intensitymix"] / 100.0
        self.hardness = DEFAULTS["hardness"] / 100.0
        self.peel = DEFAULTS["peel"]
        self.tlow = DEFAULTS["tlow"] / 100.0
        self.thigh = DEFAULTS["thigh"] / 100.0
        self.light_dir = light_dir_view(DEFAULTS["lightaz"], DEFAULTS["lightel"])
        self.steps = DEFAULTS["quality"]
        self.matcap_name = LIGHTINGS[0]
        # Clip plane. Logical state (azimuth/elevation/position/thickness/flip)
        # is the source of truth; the shader-ready normal/depth/thick are
        # recomputed from it. Both the controls and the keyboard/mouse slice
        # shortcuts drive the logical state.
        self.clip_active = 0
        self.clip_az = float(DEFAULTS["clipaz"])
        self.clip_el = float(DEFAULTS["clipel"])
        self.clip_pos = DEFAULTS["depth"] / 1000.0
        self.clip_thick_frac = DEFAULTS["thick"] / 1000.0
        self.clip_flip = False
        self.clip_normal = (0.0, 1.0, 0.0)
        self.clip_depth = 0.0
        self.clip_thick = 3.0
        self._recompute_clip()
        self.slice_overlay = 1
        self.slice_depth = DEFAULTS["overlaydepth"] / 100.0
        # Pure black, matching the 2-D slice canvas. The shader composites
        # partially-transparent rays against this, so it is both the clear
        # colour and the "empty space" colour.
        self._bg = (0.0, 0.0, 0.0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- public API --------------------------------------------------------

    def gl_ok(self) -> bool:
        return self._gl_ok

    def has_volume(self) -> bool:
        return self._volume is not None

    def is_rgb_volume(self) -> bool:
        """Whether the loaded volume carries colour rather than a scalar."""
        return self._volume is not None and self._volume[0].ndim == 4

    def set_volume(self, u8: np.ndarray, spacing: tuple[float, float, float]) -> None:
        """Upload a volume: ``(X, Y, Z)`` scalar or ``(X, Y, Z, 3)`` colour."""
        arr = np.ascontiguousarray(u8, dtype=np.uint8)
        if arr.ndim == 4:
            if arr.shape[3] < 3:
                raise ValueError(f"colour volume needs 3+ channels, got {arr.shape}")
            arr = np.ascontiguousarray(arr[..., :3])   # drop any alpha
        elif arr.ndim != 3:
            raise ValueError(f"volume must be 3-D or 4-D colour, got {arr.shape}")
        self._volume = (arr, spacing)
        if self._prog and self._gl_ok and self._make_current():
            try:
                self._upload_volume()
            finally:
                self.doneCurrent()
            self.update()

    def set_volume_float(self, vol, spacing) -> None:
        """Normalise and upload. ``(X, Y, Z)`` scalar or ``(X, Y, Z, 3+)`` colour.

        Colour volumes are scaled by ONE window across all channels, so the
        hue of each voxel survives; windowing per channel would recolour the
        data (a colour-FA map's direction encoding would be lost).
        """
        arr = np.asarray(vol)
        if arr.ndim == 4:
            rgb = arr[..., :3].astype(np.float32)
            hi = float(np.nanmax(rgb)) if rgb.size else 0.0
            if not np.isfinite(hi) or hi <= 0:
                hi = 1.0
            scale = 255.0 if hi <= 1.0 + 1e-6 else 255.0 / hi
            u8 = np.clip(np.nan_to_num(rgb) * scale, 0, 255).astype(np.uint8)
            self.set_volume(u8, spacing)
            return
        self.set_volume(normalize_to_u8(arr), spacing)

    def set_matcap(self, name: str) -> None:
        if name not in _LIGHTINGS:
            return
        self.matcap_name = name
        self._matcap_dirty = True
        if self._prog and self._gl_ok and self._make_current():
            try:
                self._refresh_matcap()
            finally:
                self.doneCurrent()
            self.update()

    def set_show_cube(self, on: bool) -> None:
        self._show_cube = bool(on)
        self.update()

    def set_flip(self, fx: float, fy: float, fz: float) -> None:
        """Mirror the render along the canonical RAS axes (L/R, A/P, S/I).

        Each argument is +1 (keep) or -1 (flip). Drives the volume mirror and
        the orientation-cube letter swap so the 3-D view matches the 2-D
        slices under the RAS / Radiological toggles.
        """
        self._flip = np.array(
            [1.0 if fx >= 0 else -1.0,
             1.0 if fy >= 0 else -1.0,
             1.0 if fz >= 0 else -1.0], dtype=np.float32,
        )
        self.update()

    def refresh(self) -> None:
        self._matcap_dirty = True
        if self._prog and self._gl_ok and self._make_current():
            try:
                self._refresh_matcap()
                if self._volume is not None:
                    self._upload_volume()
            finally:
                self.doneCurrent()
        self.update()

    def clear(self) -> None:
        self._volume = None
        if self._tex and self._make_current():
            try:
                self._safe(lambda: GL.glDeleteTextures([self._tex]))
                self._tex = 0
            finally:
                self.doneCurrent()
            self.update()

    def reset_view(self) -> None:
        self._az, self._el, self._dist = -0.6, 0.3, 1.9
        self._target = np.zeros(3, np.float32)
        self.update()

    # -- clip plane (logical state -> shader uniforms) ---------------------

    def _recompute_clip(self) -> None:
        n = clip_normal_from(self.clip_az, self.clip_el)
        if self.clip_flip:
            n = (-n[0], -n[1], -n[2])
        self.clip_normal = n
        self.clip_depth = (self.clip_pos - 0.5) * 1.8
        self.clip_thick = 0.02 + self.clip_thick_frac * 2.98

    def toggle_clip(self) -> None:
        """Shift+Y — activate / deactivate the slicer."""
        self.clip_active = 0 if self.clip_active else 1
        self.clip_changed.emit()
        self.update()

    def set_clip_axis(self, az: float, el: float) -> None:
        """Shift+A/S/C — snap the cut to an anatomical plane (and enable it)."""
        self.clip_active = 1
        self.clip_flip = False
        self.clip_az, self.clip_el = float(az), float(el)
        self._recompute_clip()
        self.clip_changed.emit()
        self.update()

    def invert_clip(self) -> None:
        """Shift+X — invert (flip) the cut direction."""
        self.clip_active = 1
        self.clip_flip = not self.clip_flip
        self._recompute_clip()
        self.clip_changed.emit()
        self.update()

    def nudge_clip_pos(self, delta: float) -> None:
        """Shift+wheel — move the cut plane through the volume."""
        self.clip_pos = float(np.clip(self.clip_pos + delta, 0.0, 1.0))
        self._recompute_clip()
        self.clip_changed.emit()
        self.update()

    def drag_clip_azel(self, d_az: float, d_el: float) -> None:
        """Shift+drag — orient the cut plane freely."""
        self.clip_az = (self.clip_az + d_az) % 360.0
        self.clip_el = float(np.clip(self.clip_el + d_el, -90.0, 90.0))
        self._recompute_clip()
        self.clip_changed.emit()
        self.update()

    # -- GL lifecycle ------------------------------------------------------

    def initializeGL(self) -> None:
        self._gl_ok = False
        self._tex = 0
        try:
            self._import_gl()
            self._prog = self._build_program(_VERT, _FRAG)
            self._vao = GL.glGenVertexArrays(1)
            self._matcap_tex = GL.glGenTextures(1)
            self._matcap_dirty = True
            self._refresh_matcap()
            self._init_cube()
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glClearColor(*self._bg, 1.0)
            self._gl_ok = True
            if self._volume is not None:
                self._upload_volume()
        except Exception as exc:  # noqa: BLE001
            self._gl_ok = False
            self._gl_error = str(exc)
            log.warning("3-D GL initialisation failed: %s", exc)

    def resizeGL(self, w: int, h: int) -> None:
        if self._gl_ok:
            GL.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        if not self._gl_ok:
            return
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        dpr = self.devicePixelRatioF()
        GL.glViewport(0, 0, int(w * dpr), int(h * dpr))
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClearColor(*self._bg, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self._tex:
            return

        eye = self._eye()
        view = _look_at(eye, self._target, np.array([0, 0, 1], np.float32))
        proj = _perspective(45.0, w / h, 0.01, 20.0)
        # Mirror the volume along the flipped canonical axes by folding a
        # diagonal reflection into the view. Rays are cast in the (axis-aligned)
        # box space, so sampling and the lit normal both reflect consistently.
        r4 = np.eye(4, dtype=np.float32)
        r4[0, 0], r4[1, 1], r4[2, 2] = self._flip
        vm = view @ r4
        inv_vp = np.linalg.inv(proj @ vm).astype(np.float32)
        rot = np.ascontiguousarray(vm[:3, :3], np.float32)
        # The cube is NOT reflected (that mirrors the glyphs); its atlas letters
        # are swapped instead, so pass the un-flipped camera rotation.
        cube_rot = np.ascontiguousarray(view[:3, :3], np.float32)

        GL.glUseProgram(self._prog)
        self._set_mat4(self._prog, "uInvViewProj", inv_vp)
        self._set_mat3(self._prog, "uNormalMatrix", rot)
        self._set_vec3(self._prog, "uBoxHalf", self._box_half)
        self._set_vec3(self._prog, "uTexSize", np.array(self._tex_dims, np.float32))
        self._set_vec3(self._prog, "uBg", np.array(self._bg, np.float32))
        self._set_vec3(self._prog, "uLightDir", np.array(self.light_dir, np.float32))
        u = lambda n: GL.glGetUniformLocation(self._prog, n)
        GL.glUniform1i(u("uEffect"), int(self.effect))
        GL.glUniform1i(u("uIsRGB"), 1 if self.is_rgb_volume() else 0)
        GL.glUniform1f(u("uThreshLo"), float(self.thresh_lo))
        GL.glUniform1f(u("uThreshHi"), float(self.thresh_hi))
        GL.glUniform1f(u("uDensity"), float(self.density))
        GL.glUniform1f(u("uBrighten"), float(self.brighten))
        GL.glUniform1f(u("uSurface"), float(self.surface))
        GL.glUniform1f(u("uAmbient"), float(self.ambient))
        GL.glUniform1f(u("uDiffuse"), float(self.diffuse))
        GL.glUniform1f(u("uSpecular"), float(self.specular))
        GL.glUniform1f(u("uShininess"), float(self.shininess))
        GL.glUniform1f(u("uBoundThresh"), float(self.bound_thresh))
        GL.glUniform1f(u("uEdgeThresh"), float(self.edge_thresh))
        GL.glUniform1f(u("uEdgeMix"), float(self.edge_mix))
        GL.glUniform1f(u("uColorTemp"), float(self.color_temp))
        GL.glUniform1f(u("uGradientMix"), float(self.gradient_mix))
        GL.glUniform1f(u("uIntensityMix"), float(self.intensity_mix))
        GL.glUniform1f(u("uHardness"), float(self.hardness))
        GL.glUniform1i(u("uPeel"), int(self.peel))
        GL.glUniform1f(u("uTlow"), float(self.tlow))
        GL.glUniform1f(u("uThigh"), float(self.thigh))
        GL.glUniform1i(u("uSteps"), int(self.steps))
        GL.glUniform1i(u("uClipActive"), int(self.clip_active))
        self._set_vec3(self._prog, "uClipNormal", np.array(self.clip_normal, np.float32))
        GL.glUniform1f(u("uClipDepth"), float(self.clip_depth))
        GL.glUniform1f(u("uClipThick"), float(self.clip_thick))
        GL.glUniform1i(u("uSliceOverlay"), int(self.slice_overlay))
        GL.glUniform1f(u("uSliceDepth"), float(self.slice_depth))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._tex)
        GL.glUniform1i(u("uVol"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glUniform1i(u("uMatcap"), 1)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

        if self._show_cube and self._cube_prog:
            self._draw_cube(cube_rot, dpr)

    def _draw_cube(self, rot: np.ndarray, dpr: float) -> None:
        flip = tuple(float(v) for v in self._flip)
        if flip != self._cube_flip and self._cube_tex:
            atlas = _make_cube_atlas(96, flip)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._cube_tex)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                            atlas.shape[1], atlas.shape[0], 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                            np.ascontiguousarray(atlas))
            self._cube_flip = flip
        s = int(96 * dpr)
        GL.glViewport(int(8 * dpr), int(8 * dpr), s, s)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self._cube_prog)
        proj = _ortho(1.0, 1.0, -4.0, 4.0)
        self._set_mat4(self._cube_prog, "uProj", proj)
        self._set_mat3(self._cube_prog, "uRot", rot)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cube_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._cube_prog, "uAtlas"), 0)
        GL.glBindVertexArray(self._cube_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._cube_nverts)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _import_gl() -> None:
        global GL
        from OpenGL import GL as _GL
        GL = _GL

    def _make_current(self) -> bool:
        try:
            self.makeCurrent()
            return self.context() is not None and self.context().isValid()
        except Exception:  # noqa: BLE001
            return False

    def _safe(self, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            log.debug("GL call failed: %s", exc)
            return None

    def _eye(self) -> np.ndarray:
        ce = np.cos(self._el)
        d = np.array([ce*np.sin(self._az), ce*np.cos(self._az), np.sin(self._el)])
        return (self._target + self._dist * d).astype(np.float32)

    def _camera_basis(self):
        eye = self._eye()
        up = np.array([0, 0, 1], np.float32)
        f = _normalize(self._target - eye)
        r = _normalize(np.cross(f, up))
        return r, np.cross(r, f)

    def _init_cube(self) -> None:
        self._cube_prog = self._build_program(_CUBE_VERT, _CUBE_FRAG)
        geo = _cube_geometry()
        self._cube_nverts = geo.shape[0]
        self._cube_vao = GL.glGenVertexArrays(1)
        self._cube_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._cube_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._cube_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, geo.nbytes, geo, GL.GL_STATIC_DRAW)
        stride = 8 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(24))
        GL.glBindVertexArray(0)
        atlas = _make_cube_atlas(96)
        self._cube_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cube_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, atlas.shape[1], atlas.shape[0],
                        0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(atlas))

    def _refresh_matcap(self) -> None:
        if not self._matcap_dirty or not self._matcap_tex:
            return
        mc = self._matcaps.get(self.matcap_name)
        if mc is None:
            mc = _make_matcap(self.matcap_name, 256)
            self._matcaps[self.matcap_name] = mc
        h, w = mc.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(mc))
        self._matcap_dirty = False

    def _upload_volume(self) -> None:
        if self._volume is None:
            return
        u8, spacing = self._volume
        # (X, Y, Z) scalar, or (X, Y, Z, 3) colour (colour-FA and friends).
        rgb = u8.ndim == 4
        x, y, z = u8.shape[:3]
        self._tex_dims = (x, y, z)
        extent = np.array([x*spacing[0], y*spacing[1], z*spacing[2]], np.float32)
        self._box_half = (0.5 * extent / float(extent.max())).astype(np.float32)
        data = np.ascontiguousarray(
            u8.transpose(2, 1, 0, 3) if rgb else u8.transpose(2, 1, 0)
        )
        if self._tex:
            GL.glDeleteTextures([self._tex])
        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        for pn in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R):
            GL.glTexParameteri(GL.GL_TEXTURE_3D, pn, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        if rgb:
            GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB8, x, y, z, 0,
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data)
        else:
            GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R8, x, y, z, 0,
                            GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)

    def _build_program(self, vsrc: str, fsrc: str) -> int:
        def compile_stage(src, stage):
            sh = GL.glCreateShader(stage)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(sh).decode())
            return sh
        vs = compile_stage(vsrc, GL.GL_VERTEX_SHADER)
        fs = compile_stage(fsrc, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def _set_mat4(self, prog, name, m):
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(prog, name), 1, GL.GL_TRUE,
                              np.ascontiguousarray(m, np.float32))

    def _set_mat3(self, prog, name, m):
        GL.glUniformMatrix3fv(GL.glGetUniformLocation(prog, name), 1, GL.GL_TRUE,
                              np.ascontiguousarray(m, np.float32))

    def _set_vec3(self, prog, name, v):
        GL.glUniform3f(GL.glGetUniformLocation(prog, name),
                       float(v[0]), float(v[1]), float(v[2]))

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._last = ev.position().toPoint()
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift and self.clip_active:
            self._drag = "clip"          # Shift-drag orients the cut plane
        elif ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag = "pan"
        else:
            self._drag = "rotate"

    def mouseMoveEvent(self, ev) -> None:
        if self._last is None:
            return
        p = ev.position().toPoint()
        dx = p.x() - self._last.x()
        dy = p.y() - self._last.y()
        self._last = p
        if self._drag == "clip":
            self.drag_clip_azel(dx * 0.6, -dy * 0.6)   # emits + updates
            return
        if self._drag == "pan":
            r, u = self._camera_basis()
            scale = 0.0022 * self._dist
            self._target += (-dx * r + dy * u) * scale
        else:
            self._az += dx * 0.01
            self._el = float(np.clip(self._el + dy * 0.01, -1.55, 1.55))
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._last = None

    def wheelEvent(self, ev) -> None:
        ad, pd = ev.angleDelta(), ev.pixelDelta()
        # Magnitude in "notches": a mouse-wheel notch is ±1, a trackpad's many
        # tiny events are fractional — so zoom and slice-nav both scale with it
        # and neither is hyper-sensitive.
        a = ad.y() or ad.x()
        steps = (a / 120.0) if a != 0 else ((pd.y() or pd.x()) / 320.0)
        if steps == 0.0:
            return
        # Slice navigator when the clip plane is active. Shift+wheel is the
        # gesture, but X11 / macOS commonly remap Shift+vertical-wheel to a
        # *horizontal* scroll and drop the Shift flag — so a purely horizontal
        # scroll over the render also navigates the slice.
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        horizontal = ((ad.x() != 0 and ad.y() == 0)
                      or (pd.x() != 0 and pd.y() == 0))
        if self.clip_active and (shift or horizontal):
            self.nudge_clip_pos(0.02 * steps)            # slice navigator
            return
        self._dist = float(np.clip(self._dist * (0.9 ** steps), 0.5, 12.0))  # zoom
        self.update()


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

_THRESH_GAP = 20


class Nifti3DControls(QWidget):
    """Effect / lighting / transfer-function / clip controls, one per render.

    Vertical column (used in a scroll area). Per-effect the irrelevant
    parameters grey out (:data:`EFFECT_PARAMS`), mirroring MRIcroGL.
    """

    def __init__(self, gl: RaycastGLWidget, *, vertical: bool = True, parent=None) -> None:
        super().__init__(parent)
        # Own identity (it used to borrow "sidecar-toolbar", a QFrame rule that
        # never matched this QWidget — so the panel painted nothing and the
        # black canvas behind showed through).
        self.setObjectName("nifti-3d-controls")
        self.gl = gl
        self._vertical = vertical
        self._rows: dict[str, QWidget] = {}     # key -> label+widget container
        # Slider value read-out: (slider, chip-label, base-text) per slider row,
        # so "Show values" can append the live integer to each label. The raw
        # slider units are exactly what the presets in EFFECT_PRESET use.
        self._value_rows: list = []
        self._show_values = False
        # Every effect keeps its OWN parameter values: label -> {key: value}.
        # Switching effects stashes the outgoing set and restores the incoming
        # one, so tweaks to one effect never leak into another.
        self._effect_values: dict[str, dict] = {}
        self._current_effect: Optional[str] = None
        self._build_widgets()
        self._layout()
        self._apply_clip()
        self._on_effect(self._effect.currentIndex())   # set shader fx + graying
        # Keyboard / Shift-drag changes to the clip on the GL widget mirror back
        # into these sliders.
        self.gl.clip_changed.connect(self._sync_clip_from_gl)

    def _build_widgets(self) -> None:
        self._effect = QComboBox(); self._effect.addItems(EFFECTS)
        self._effect.currentIndexChanged.connect(self._on_effect)
        self._light = QComboBox(); self._light.addItems(LIGHTINGS)
        self._light.currentIndexChanged.connect(self._on_light)

        self._lo = self._sl(0, 1000, DEFAULTS["lo"], self._on_lo)
        self._hi = self._sl(0, 1000, DEFAULTS["hi"], self._on_hi)
        self._den = self._sl(0, 300, DEFAULTS["density"], self._on_den)
        self._bright = self._sl(50, 300, DEFAULTS["brighten"], self._on_bright)
        self._surf = self._sl(0, 100, DEFAULTS["surface"], self._on_surf)
        self._amb = self._sl(0, 150, DEFAULTS["ambient"], self._on_amb)
        self._dif = self._sl(0, 150, DEFAULTS["diffuse"], self._on_dif)
        self._spec = self._sl(0, 100, DEFAULTS["specular"], self._on_spec)
        self._shin = self._sl(1, 100, DEFAULTS["shininess"], self._on_shin)
        self._bound = self._sl(0, 100, DEFAULTS["boundthresh"], self._on_bound)
        self._edge = self._sl(0, 100, DEFAULTS["edgethresh"], self._on_edge)
        self._emix = self._sl(0, 100, DEFAULTS["edgemix"], self._on_emix)
        self._ctemp = self._sl(0, 100, DEFAULTS["colortemp"], self._on_ctemp)
        self._gmix = self._sl(0, 100, DEFAULTS["gradientmix"], self._on_gmix)
        self._imix = self._sl(0, 100, DEFAULTS["intensitymix"], self._on_imix)
        self._hard = self._sl(0, 100, DEFAULTS["hardness"], self._on_hard)
        self._peel = self._sl(0, 6, DEFAULTS["peel"], self._on_peel)
        self._tlow = self._sl(0, 100, DEFAULTS["tlow"], self._on_tlow)
        self._thigh = self._sl(0, 100, DEFAULTS["thigh"], self._on_thigh)
        self._q = self._sl(QUALITY_MIN, QUALITY_MAX, DEFAULTS["quality"], self._on_q)
        self._laz = self._sl(0, 360, DEFAULTS["lightaz"], self._on_light_dir)
        self._lel = self._sl(-90, 90, DEFAULTS["lightel"], self._on_light_dir)
        self._overlay_en = QCheckBox("Slice overlay")
        self._overlay_en.setChecked(bool(DEFAULTS["overlay"]))
        self._overlay_en.stateChanged.connect(self._on_overlay)
        self._odepth = self._sl(0, 100, DEFAULTS["overlaydepth"], self._on_odepth)

        self._clip_en = QCheckBox("Enable clip plane")
        self._clip_en.stateChanged.connect(self._on_clip)
        self._caz = self._sl(0, 360, DEFAULTS["clipaz"], self._on_clip)
        self._cel = self._sl(-90, 90, DEFAULTS["clipel"], self._on_clip)
        self._cdepth = self._sl(0, 1000, DEFAULTS["depth"], self._on_clip)
        self._cthick = self._sl(0, 1000, DEFAULTS["thick"], self._on_clip)

        self._values_en = QCheckBox("Show values")
        self._values_en.setChecked(self._show_values)
        self._values_en.setToolTip(
            "Append each slider's current value to its label — read off exact "
            "numbers to build presets."
        )
        self._values_en.stateChanged.connect(self._on_show_values)

        self._refresh_btn = self._btn("Refresh image", lambda: self.gl.refresh())
        self._reset_params_btn = self._btn("Reset parameters", self.reset_params)
        self._reset_params_btn.setToolTip(
            "Reset THIS effect's parameters to its defaults (or its preset). "
            "Other effects keep their own values."
        )
        self._reset_all_btn = self._btn("Reset all effects", self.reset_all_params)
        self._reset_all_btn.setToolTip(
            "Reset the parameters of EVERY effect back to their defaults / presets."
        )
        self._reset_view_btn = self._btn("Reset view", self.gl.reset_view)

    def _layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        outer.addWidget(self._values_en)
        specs = [
            ("effect", "Effect", self._effect), ("light", "Lighting", self._light),
            ("lo", "Threshold low", self._lo), ("hi", "Threshold high", self._hi),
            ("density", "Density", self._den),
            ("brighten", "Brightness", self._bright), ("surface", "Surface colour", self._surf),
            ("ambient", "Ambient", self._amb), ("diffuse", "Diffuse", self._dif),
            ("specular", "Specular", self._spec), ("shininess", "Shininess", self._shin),
            ("boundthresh", "Boundary thresh", self._bound),
            ("edgethresh", "Edge thresh", self._edge), ("edgemix", "Edge/bound mix", self._emix),
            ("colortemp", "Colour temp", self._ctemp),
            ("gradientmix", "Gradient mix", self._gmix),
            ("intensitymix", "Intensity mix", self._imix), ("hardness", "Surface hardness", self._hard),
            ("peel", "Peel layers", self._peel), ("tlow", "T low", self._tlow),
            ("thigh", "T high", self._thigh),
            ("lightaz", "Light azimuth", self._laz), ("lightel", "Light elevation", self._lel),
            ("overlay", None, self._overlay_en), ("overlaydepth", "Overlay depth", self._odepth),
            ("quality", "Quality", self._q),
        ]
        for key, lbl, wdg in specs:
            row = self._stacked(lbl, wdg) if lbl is not None else self._wrap(wdg)
            self._rows[key] = row
            outer.addWidget(row)

        outer.addSpacing(4)
        outer.addWidget(self._clip_en)
        for lbl, wdg in (("Clip azimuth", self._caz), ("Clip elevation", self._cel),
                         ("Clip depth", self._cdepth), ("Clip thickness", self._cthick)):
            outer.addWidget(self._stacked(lbl, wdg))
        outer.addSpacing(6)
        for b in (self._refresh_btn, self._reset_params_btn,
                  self._reset_all_btn, self._reset_view_btn):
            outer.addWidget(b)
        outer.addStretch(1)
        hint = self._chip("Left-drag rotate · right-drag pan · scroll zoom · O = cube")
        hint.setWordWrap(True)
        outer.addWidget(hint)

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _chip(text):
        lbl = QLabel(text); lbl.setObjectName("sidecar-footer-summary"); return lbl

    def _stacked(self, text, wdg):
        w = QWidget()
        box = QVBoxLayout(w)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        chip = self._chip(text)
        box.addWidget(chip)
        box.addWidget(wdg)
        # Register sliders so "Show values" can live-append their value.
        if isinstance(wdg, QSlider):
            self._value_rows.append((wdg, chip, text))
            wdg.valueChanged.connect(
                lambda _v, c=chip, b=text, s=wdg: self._update_value_label(c, b, s)
            )
        return w

    def _update_value_label(self, chip, base, slider) -> None:
        chip.setText(f"{base}   {slider.value()}" if self._show_values else base)

    def _on_show_values(self) -> None:
        self._show_values = self._values_en.isChecked()
        for slider, chip, base in self._value_rows:
            self._update_value_label(chip, base, slider)

    @staticmethod
    def _wrap(wdg):
        w = QWidget()
        box = QVBoxLayout(w)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(wdg)
        return w

    @staticmethod
    def _sl(lo, hi, val, cb):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(180)
        s.valueChanged.connect(cb)
        return s

    @staticmethod
    def _btn(text, cb):
        b = QPushButton(text); b.setObjectName("tb-btn-toggle"); b.clicked.connect(cb)
        return b

    # -- slots -------------------------------------------------------------

    def _on_effect(self, i):
        """Switch effects, keeping every effect's parameters independent.

        The outgoing effect's slider values are stashed and the incoming one's
        are restored, so tweaking (say) Jelly never disturbs Skull. An effect
        seen for the first time starts from its baseline — its preset if it has
        one, otherwise the global defaults.
        """
        label = EFFECTS[int(i)]
        if self._current_effect is not None and self._current_effect != label:
            self._effect_values[self._current_effect] = self._capture_values()
        self._current_effect = label
        self.gl.effect = EFFECT_FX[label]         # display label -> shader fx
        self._sync_effect_params()
        self._apply_values(self._stored_values(label))
        # setChecked/setValue only signal on a *change*, so push the overlay
        # state explicitly — the incoming effect may share the outgoing one's
        # checkbox state while the GL still holds a stale value.
        self.gl.slice_overlay = 1 if self._overlay_en.isChecked() else 0
        self.gl.update()

    def _apply_values(self, values: dict) -> None:
        """Push control values into the widgets (which drive the GL)."""
        widgets = self._param_widgets()
        for key, val in values.items():
            if key in widgets:
                self._widget_set(widgets[key][0], val)

    def _sync_effect_params(self) -> None:
        """Show only the rows this effect uses. The slice-overlay state itself
        is per-effect parameter state (its baseline is :data:`SLICE_DEFAULT_ON`)
        and is restored by :meth:`_apply_values`."""
        name = EFFECTS[self._effect.currentIndex()]
        keys = EFFECT_PARAMS[name]
        for key, row in self._rows.items():
            if key == "effect":
                continue
            row.setVisible(key in keys)

    def _on_light(self, i): self.gl.set_matcap(LIGHTINGS[int(i)])

    def _on_lo(self, v):
        if v > self._hi.value() - _THRESH_GAP:
            self._hi.blockSignals(True); self._hi.setValue(min(1000, v + _THRESH_GAP)); self._hi.blockSignals(False)
            self.gl.thresh_hi = self._hi.value() / 1000.0
        self.gl.thresh_lo = v / 1000.0; self.gl.update()

    def _on_hi(self, v):
        if v < self._lo.value() + _THRESH_GAP:
            self._lo.blockSignals(True); self._lo.setValue(max(0, v - _THRESH_GAP)); self._lo.blockSignals(False)
            self.gl.thresh_lo = self._lo.value() / 1000.0
        self.gl.thresh_hi = v / 1000.0; self.gl.update()

    def _on_den(self, v): self.gl.density = v / 100.0; self.gl.update()
    def _on_bright(self, v): self.gl.brighten = v / 100.0; self.gl.update()
    def _on_surf(self, v): self.gl.surface = v / 100.0; self.gl.update()
    def _on_amb(self, v): self.gl.ambient = v / 100.0; self.gl.update()
    def _on_dif(self, v): self.gl.diffuse = v / 100.0; self.gl.update()
    def _on_spec(self, v): self.gl.specular = v / 100.0; self.gl.update()
    def _on_shin(self, v): self.gl.shininess = float(v); self.gl.update()
    def _on_bound(self, v): self.gl.bound_thresh = v / 100.0; self.gl.update()
    def _on_edge(self, v): self.gl.edge_thresh = v / 100.0; self.gl.update()
    def _on_emix(self, v): self.gl.edge_mix = v / 100.0; self.gl.update()
    def _on_ctemp(self, v): self.gl.color_temp = v / 100.0; self.gl.update()
    def _on_gmix(self, v): self.gl.gradient_mix = v / 100.0; self.gl.update()
    def _on_imix(self, v): self.gl.intensity_mix = v / 100.0; self.gl.update()
    def _on_hard(self, v): self.gl.hardness = v / 100.0; self.gl.update()
    def _on_peel(self, v): self.gl.peel = int(v); self.gl.update()
    def _on_tlow(self, v): self.gl.tlow = v / 100.0; self.gl.update()
    def _on_thigh(self, v): self.gl.thigh = v / 100.0; self.gl.update()
    def _on_q(self, v): self.gl.steps = int(v); self.gl.update()
    def _on_overlay(self, *_a): self.gl.slice_overlay = 1 if self._overlay_en.isChecked() else 0; self.gl.update()
    def _on_odepth(self, v): self.gl.slice_depth = v / 100.0; self.gl.update()

    def _on_light_dir(self, *_a):
        self.gl.light_dir = light_dir_view(self._laz.value(), self._lel.value())
        self.gl.update()

    def _on_clip(self, *_a): self._apply_clip()

    def _apply_clip(self) -> None:
        on = self._clip_en.isChecked()
        for w in (self._caz, self._cel, self._cdepth, self._cthick):
            w.setEnabled(on)
        self.gl.clip_active = 1 if on else 0
        self.gl.clip_az = float(self._caz.value())
        self.gl.clip_el = float(self._cel.value())
        self.gl.clip_pos = self._cdepth.value() / 1000.0
        self.gl.clip_thick_frac = self._cthick.value() / 1000.0
        self.gl._recompute_clip()
        self.gl.update()

    def _sync_clip_from_gl(self) -> None:
        """Mirror the clip sliders after a keyboard/Shift-drag change on the GL."""
        pairs = (
            (self._caz, int(round(self.gl.clip_az))),
            (self._cel, int(round(self.gl.clip_el))),
            (self._cdepth, int(round(self.gl.clip_pos * 1000))),
            (self._cthick, int(round(self.gl.clip_thick_frac * 1000))),
        )
        for w, val in pairs:
            was = w.blockSignals(True)
            w.setValue(val)
            w.setEnabled(bool(self.gl.clip_active))
            w.blockSignals(was)
        was = self._clip_en.blockSignals(True)
        self._clip_en.setChecked(bool(self.gl.clip_active))
        self._clip_en.blockSignals(was)

    # -- keyboard slicer shortcuts (driven from the pane) ------------------

    def kbd_toggle_clip(self) -> None:
        self._clip_en.toggle()          # Shift+Y — drives _apply_clip

    def kbd_set_axis(self, az: int, el: int) -> None:
        self._clip_en.setChecked(True)  # Shift+A/S/C
        self._caz.setValue(int(az))
        self._cel.setValue(int(el))

    def kbd_invert(self) -> None:
        if not self._clip_en.isChecked():
            self._clip_en.setChecked(True)
        self.gl.invert_clip()           # Shift+X

    # -- per-effect parameter state ----------------------------------------

    def _param_widgets(self):
        """Control key -> (widget, default). Single source of truth for the
        per-effect parameter state: getters, setters and defaults derive from
        it, so a new parameter only has to be registered once."""
        return {
            "light": (self._light, DEFAULTS["light"]),
            "lo": (self._lo, DEFAULTS["lo"]),
            "hi": (self._hi, DEFAULTS["hi"]),
            "density": (self._den, DEFAULTS["density"]),
            "brighten": (self._bright, DEFAULTS["brighten"]),
            "surface": (self._surf, DEFAULTS["surface"]),
            "ambient": (self._amb, DEFAULTS["ambient"]),
            "diffuse": (self._dif, DEFAULTS["diffuse"]),
            "specular": (self._spec, DEFAULTS["specular"]),
            "shininess": (self._shin, DEFAULTS["shininess"]),
            "boundthresh": (self._bound, DEFAULTS["boundthresh"]),
            "edgethresh": (self._edge, DEFAULTS["edgethresh"]),
            "edgemix": (self._emix, DEFAULTS["edgemix"]),
            "colortemp": (self._ctemp, DEFAULTS["colortemp"]),
            "gradientmix": (self._gmix, DEFAULTS["gradientmix"]),
            "intensitymix": (self._imix, DEFAULTS["intensitymix"]),
            "hardness": (self._hard, DEFAULTS["hardness"]),
            "peel": (self._peel, DEFAULTS["peel"]),
            "tlow": (self._tlow, DEFAULTS["tlow"]),
            "thigh": (self._thigh, DEFAULTS["thigh"]),
            "lightaz": (self._laz, DEFAULTS["lightaz"]),
            "lightel": (self._lel, DEFAULTS["lightel"]),
            "overlay": (self._overlay_en, DEFAULTS["overlay"]),
            "overlaydepth": (self._odepth, DEFAULTS["overlaydepth"]),
            "quality": (self._q, DEFAULTS["quality"]),
        }

    @staticmethod
    def _widget_get(w):
        if isinstance(w, QSlider):
            return w.value()
        if isinstance(w, QComboBox):
            return w.currentIndex()
        return w.isChecked()            # QCheckBox

    @staticmethod
    def _widget_set(w, val) -> None:
        if isinstance(w, QSlider):
            w.setValue(int(val))
        elif isinstance(w, QComboBox):
            w.setCurrentIndex(int(val))
        else:
            w.setChecked(bool(val))     # QCheckBox

    def _baseline_values(self, label: str) -> dict:
        """Every parameter's starting value for ``label`` — its preset value if
        the effect defines one, the per-effect slice-overlay default for
        ``overlay``, otherwise the global default."""
        preset = EFFECT_PRESET.get(label) or {}
        vals = {}
        for key, (_w, default) in self._param_widgets().items():
            if key in preset:
                vals[key] = preset[key]
            elif key == "overlay":
                vals[key] = label in SLICE_DEFAULT_ON
            else:
                vals[key] = default
        return vals

    def _capture_values(self) -> dict:
        """Snapshot the current widget values (the active effect's state)."""
        return {key: self._widget_get(w)
                for key, (w, _d) in self._param_widgets().items()}

    def _stored_values(self, label: str) -> dict:
        """The remembered values for ``label``, seeded from its baseline."""
        vals = self._effect_values.get(label)
        if vals is None:
            vals = self._baseline_values(label)
            self._effect_values[label] = vals
        return vals

    # Control key -> (widget setter, default value). Derived from
    # :meth:`_param_widgets` so the registry lives in exactly one place.
    def _param_setters(self):
        return {
            key: ((lambda v, _w=w: self._widget_set(_w, v)), default)
            for key, (w, default) in self._param_widgets().items()
        }

    def reset_params(self) -> None:
        """Reset ONLY the parameters that belong to the current effect back to
        their defaults (or, for a preset effect like Jelly/Skull, back to the
        preset). Other effects keep their own values, and the effect itself,
        the clip plane and the camera are left as they are (there is a separate
        Reset view button)."""
        label = EFFECTS[self._effect.currentIndex()]
        baseline = self._baseline_values(label)
        stored = self._stored_values(label)
        widgets = self._param_widgets()
        for key in EFFECT_PARAMS[label]:
            if key in baseline:
                stored[key] = baseline[key]
                self._widget_set(widgets[key][0], baseline[key])

    def reset_all_params(self) -> None:
        """Reset EVERY effect's parameters back to its baseline, then re-apply
        the current effect's. Use when the per-effect tweaks have drifted and
        you want a clean slate across the whole effect list."""
        self._effect_values = {label: self._baseline_values(label)
                               for label in EFFECTS}
        label = EFFECTS[self._effect.currentIndex()]
        self._apply_values(self._effect_values[label])


# --------------------------------------------------------------------------
# Composite view (pure-3-D mode)
# --------------------------------------------------------------------------

class Nifti3DView(QWidget):
    """GL canvas + a scrollable vertical controls panel (the pane's "3D" mode)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")
        self.gl = RaycastGLWidget()
        self.controls = Nifti3DControls(self.gl, vertical=True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.controls)
        scroll.setFixedWidth(224)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self.gl, 1)
        root.addWidget(scroll)

    def set_volume(self, vol, spacing): self.gl.set_volume_float(vol, spacing)
    def clear(self): self.gl.clear()
    def reset_view(self): self.gl.reset_view()
    def set_show_cube(self, on): self.gl.set_show_cube(on)


__all__ = [
    "Nifti3DControls", "Nifti3DView", "RaycastGLWidget", "request_gl_format",
    "gpu_available", "normalize_to_u8", "clip_normal_from", "light_dir_view",
    "EFFECTS", "LIGHTINGS", "EFFECT_PARAMS", "DEFAULTS",
]
