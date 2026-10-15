"""The 3-D render's vocabulary and maths, without Qt or OpenGL.

Effects, lighting materials, the parameters each effect uses with their
ranges and presets, and the camera / clip-plane / matcap maths. The GL canvas
(``bidsmgr/gui/viz/canvases/render.py``) only uploads and draws; everything a
scene can SAY about a render is defined here, so the scene can name an effect,
a control panel can be generated from :data:`PARAMS`, and two viewers can
compare render states without a GL context.

Parameter values are kept in the slider units the presets were written in
(``lo=36`` means 0.036 of the normalised range), because those are the
numbers users read off the panel and type into presets. :func:`uniform_values`
converts them to what the shader takes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# -- effects ------------------------------------------------------------------

#: Effect labels in menu order. Several are named presets of one shader path.
EFFECTS = ["Standard", "Matte", "Juicy shiny", "Juicy shiny 2", "Realistic",
           "Glass", "X-ray", "Jelly", "Skull", "MIP", "Edges",
           "Opacity peeling", "Opacity peeling 2", "Shell", "Topography"]

#: Label -> shader effect index (``uEffect``).
EFFECT_FX: dict[str, int] = {
    "Standard": 1, "Matte": 0, "Juicy shiny": 1, "Juicy shiny 2": 1,
    "Realistic": 10,
    "Glass": 2, "X-ray": 3, "MIP": 4, "Edges": 5,
    "Opacity peeling": 6, "Opacity peeling 2": 7, "Shell": 8, "Topography": 9,
    "Jelly": 6, "Skull": 6,
}

_COMMON = {"lo", "hi", "quality", "layers", "seethrough"}
_OVERLAY = {"overlay", "overlaydepth"}
_MATCAP = _COMMON | _OVERLAY | {"density", "light", "brighten", "surface",
                                "ambient", "diffuse", "specular",
                                "shininess", "lightaz", "lightel"}
_PHONG = _COMMON | _OVERLAY | {"density", "ambient", "diffuse", "specular",
                               "shininess", "lightaz", "lightel"}
_PEEL = _COMMON | {"density", "ambient", "diffuse", "specular", "shininess",
                   "peel", "tlow", "thigh", "lightaz", "lightel"}

#: Which parameters each effect uses (the panel hides the rest).
EFFECT_PARAMS: dict[str, set] = {
    "Standard": _PHONG,
    "Matte": _MATCAP,
    "Juicy shiny": _PHONG,
    "Juicy shiny 2": _PHONG,
    "Realistic": _PHONG,
    "Glass": _COMMON | {"specular", "shininess", "edgethresh", "boundthresh",
                        "edgemix", "colortemp", "lightaz", "lightel"},
    "X-ray": _COMMON | {"density"},
    "MIP": _COMMON,
    "Edges": _COMMON | {"density", "light", "brighten", "surface",
                        "boundthresh", "edgethresh", "edgemix",
                        "lightaz", "lightel"},
    "Opacity peeling": _PEEL,
    "Opacity peeling 2": _PEEL,
    "Shell": _COMMON | {"boundthresh", "edgethresh", "edgemix", "colortemp",
                        "specular", "lightaz", "lightel"},
    "Topography": _COMMON | _OVERLAY | {"density", "light", "brighten",
                                        "surface", "gradientmix",
                                        "intensitymix", "hardness",
                                        "lightaz", "lightel"},
    "Jelly": _PEEL,
    "Skull": _PEEL,
}

#: Effects whose cut-face intensity slice is on by default.
SLICE_DEFAULT_ON = {"Standard", "Matte", "Juicy shiny", "Juicy shiny 2",
                    "Realistic", "Topography"}

#: Named parameter presets applied when a preset effect is first chosen.
EFFECT_PRESET: dict[str, dict] = {
    "Juicy shiny": dict(lo=36, hi=303, density=150, ambient=94, diffuse=50,
                        specular=50, shininess=20, lightaz=0, lightel=30,
                        overlay=True, overlaydepth=68),
    "Juicy shiny 2": dict(lo=36, hi=303, density=150, ambient=94, diffuse=23,
                          specular=96, shininess=100, lightaz=0, lightel=30,
                          overlay=True, overlaydepth=68),
    "Realistic": dict(lo=36, hi=303, density=150, ambient=52, diffuse=78,
                      specular=62, shininess=80, lightaz=0, lightel=30,
                      overlay=True, overlaydepth=68),
    "Jelly": dict(lo=110, hi=400, density=7, ambient=115, diffuse=65,
                  specular=35, shininess=45, peel=1, tlow=13, thigh=82,
                  lightaz=0, lightel=25),
    "Skull": dict(lo=36, hi=303, density=150, ambient=95, diffuse=50,
                  specular=50, shininess=20, peel=1, tlow=19, thigh=80,
                  lightaz=0, lightel=30),
}

# -- lighting materials (matcaps) ---------------------------------------------

#: name -> (ambient, key, fill, spec_power, spec_int, tint)
MATERIALS: dict[str, tuple] = {
    "Shiny White": (0.32, 0.90, 0.24, 55.0, 1.55, (1.00, 0.98, 0.95)),
    "Clay":        (0.38, 0.64, 0.28, 12.0, 0.16, (0.86, 0.78, 0.70)),
    "Bone":        (0.40, 0.66, 0.26, 28.0, 0.35, (0.94, 0.91, 0.84)),
    "Titanium":    (0.24, 0.82, 0.22, 96.0, 0.95, (0.72, 0.75, 0.80)),
    "Gold":        (0.30, 0.74, 0.22, 44.0, 0.85, (0.96, 0.80, 0.36)),
    "Blue":        (0.30, 0.68, 0.28, 34.0, 0.60, (0.56, 0.69, 0.96)),
}
LIGHTINGS = list(MATERIALS)

QUALITY_MIN, QUALITY_MAX = 64, 1024
#: Half the maximum: the first render is responsive on modest GPUs.
QUALITY_DEFAULT = QUALITY_MAX // 2


@dataclass(frozen=True)
class Param:
    """One render parameter as the panel shows it."""

    key: str
    label: str
    lo: int
    hi: int
    default: float
    #: Divisor from slider units to the shader's value (1 = as is).
    scale: float = 100.0
    kind: str = "slider"          # "slider" | "check" | "choice"
    help: str = ""
    #: Which section of the 3-D controls it sits in (by purpose).
    group: str = "look"


#: The sections of the 3-D controls, in panel order, with their titles.
PARAM_GROUPS: dict[str, str] = {
    "look": "3-D: appearance",
    "transfer": "3-D: opacity",
    "cut": "3-D: cut surface",
    "edges": "3-D: edges",
    "peel": "3-D: peeling",
    "lighting": "3-D: lighting",
    "overlays": "3-D: overlays",
    "quality": "3-D: rendering quality",
}

#: Every parameter, in panel order. Single source of truth for the panel,
#: the presets, the defaults and the shader conversion. Labels use the
#: field's own terms (transfer function, Phong lighting); each tooltip says
#: what the control does to the picture.
PARAMS: list[Param] = [
    Param("light", "Material", 0, len(LIGHTINGS) - 1, 0, 1, "choice",
          "The surface material (a lit sphere, or matcap) the render is shaded with."),
    Param("lo", "Opacity threshold, low", 0, 1000, 100, 1000,
          help="Transfer function: intensities below this fraction of the display range "
               "are fully transparent. Raise it to strip away background, noise and skin.",
          group="transfer"),
    Param("hi", "Opacity threshold, high", 0, 1000, 420, 1000,
          help="Transfer function: from this fraction of the display range upward tissue "
               "is fully opaque; between the low and high thresholds opacity ramps up "
               "smoothly.", group="transfer"),
    Param("density", "Absorption", 0, 300, 100,
          help="How fast opacity builds up along each ray. Low lets deeper structures "
               "show through the surface; high gives a solid surface.", group="transfer"),
    Param("brighten", "Brightness", 50, 300, 150,
          help="Overall gain of the shaded colour."),
    Param("surface", "Intensity shading", 0, 100, 70,
          help="How much the image's own intensities colour the surface: 0 is the "
               "material's colour alone, 1 the image's grey levels."),
    Param("colortemp", "Colour temperature", 0, 100, 50, group="edges",
          help="Edges and Glass effects: the outline colour, from cool (blue) to warm "
               "(orange)."),
    Param("hardness", "Relief contrast", 0, 100, 50,
          help="Topography effect: how strongly intensity differences are drawn as "
               "relief."),
    Param("gradientmix", "Gradient opacity", 0, 100, 60,
          help="Topography effect: how much opacity follows the intensity GRADIENT (tissue "
               "boundaries) rather than the intensity itself."),
    Param("intensitymix", "Intensity blend", 0, 100, 35,
          help="Topography effect: how much of the colour comes from intensity shading "
               "rather than from the material."),
    Param("overlay", "Show the slice on the cut surface", 0, 1, 1, 1, "check",
          "Where a clip plane cuts the volume, paint the cut surface with the 2-D slice "
          "(its window, colour map and overlays), as the slice views show it. Off: the "
          "rendered volume is seen through the cut.", group="cut"),
    Param("overlaydepth", "Cavity threshold", 0, 100, 28,
          help="On the cut surface, intensities below this fraction of the range are left "
               "open, so cavities (ventricles, sulci, air) appear as recesses with the "
               "rendered surface behind them.", group="cut"),
    Param("boundthresh", "Boundary threshold", 0, 100, 30, group="edges",
          help="Edges and Glass effects: the gradient strength above which a voxel counts "
               "as a tissue boundary."),
    Param("edgethresh", "Contour threshold", 0, 100, 12, group="edges",
          help="Edges effect: the gradient strength above which a thin contour is drawn."),
    Param("edgemix", "Contours against boundaries", 0, 100, 65, group="edges",
          help="0 emphasises broad tissue boundaries, 1 thin contour lines."),
    Param("peel", "Layers peeled", 0, 6, 1, 1, group="peel",
          help="Opacity peeling: how many surfaces are removed before the one shown "
               "(0: the outer surface; 1: the next one inside; and so on)."),
    Param("tlow", "Layer exit threshold", 0, 100, 25, group="peel",
          help="Peeling: once a layer is full, it ends where a sample's opacity falls "
               "below this."),
    Param("thigh", "Layer fill threshold", 0, 100, 85, group="peel",
          help="Peeling: the accumulated opacity at which a layer counts as full."),
    Param("ambient", "Ambient", 0, 150, 60, group="lighting",
          help="Phong lighting: light every surface receives, whatever way it faces."),
    Param("diffuse", "Diffuse", 0, 150, 55, group="lighting",
          help="Phong lighting: light in proportion to how directly a surface faces the "
               "light."),
    Param("specular", "Specular", 0, 100, 30, group="lighting",
          help="Phong lighting: the strength of the highlights."),
    Param("shininess", "Shininess", 1, 100, 40, 1, group="lighting",
          help="Phong lighting: how tight the highlights are (higher: smaller and "
               "sharper)."),
    Param("lightaz", "Light azimuth", 0, 360, 0, 1, group="lighting",
          help="Where the light comes from, around the viewing direction (degrees)."),
    Param("lightel", "Light elevation", -90, 90, 0, 1, group="lighting",
          help="Where the light comes from, above (positive) or below the viewing "
               "direction (degrees)."),
    Param("layers", "Show overlays in 3-D", 0, 1, 1, 1, "check",
          "Draw the overlays (atlases, maps, masks, the MRS voxel) in the render, "
          "coloured as the slices draw them.", group="overlays"),
    Param("seethrough", "Overlay see-through", 0, 100, 40,
          help="How strongly an overlay inside the head is laid over the surface in front "
               "of it. 0: depth only, it shows where the volume is cut open; 1: drawn over "
               "everything.", group="overlays"),
    Param("quality", "Ray samples", QUALITY_MIN, QUALITY_MAX, QUALITY_DEFAULT, 1,
          help="Samples taken along each ray. More shows finer detail and renders slower.",
          group="quality"),
]
PARAM_BY_KEY = {p.key: p for p in PARAMS}

#: Thresholds keep at least this gap (slider units) so the ramp never inverts.
THRESH_GAP = 20

#: Parameters about the overlays rather than the look: one value for every
#: effect (``RenderState.shared``), never reset with an effect.
SHARED_PARAMS = frozenset({"layers", "seethrough"})


def baseline(effect: str) -> dict[str, float]:
    """An effect's starting values: its preset, else the global defaults."""
    preset = EFFECT_PRESET.get(effect, {})
    out: dict[str, float] = {}
    for p in PARAMS:
        if p.key in preset:
            out[p.key] = float(preset[p.key])
        elif p.key == "overlay":
            out[p.key] = 1.0 if effect in SLICE_DEFAULT_ON else 0.0
        else:
            out[p.key] = float(p.default)
    return out


def effective(effect: str, overrides: dict[str, float] | None,
              shared: dict[str, float] | None = None) -> dict[str, float]:
    """Baseline merged with the user's changes for that effect, and with the
    shared values (:data:`SHARED_PARAMS`)."""
    out = baseline(effect)
    for key, value in (overrides or {}).items():
        if key in out and key not in SHARED_PARAMS:
            out[key] = float(value)
    for key, value in (shared or {}).items():
        if key in SHARED_PARAMS:
            out[key] = float(value)
    return out


def values_for(render) -> dict[str, float]:
    """Every parameter's value for a ``RenderState`` as it stands."""
    return effective(render.effect, render.params.get(render.effect), render.shared)


def clamp_param(key: str, value: float) -> float:
    p = PARAM_BY_KEY[key]
    return float(min(max(value, p.lo), p.hi))


def uniform_values(values: dict[str, float]) -> dict[str, float]:
    """Slider units to the numbers the shader expects."""
    out = {}
    for key, value in values.items():
        p = PARAM_BY_KEY.get(key)
        if p is None:
            continue
        out[key] = float(value) / (p.scale if p.scale else 1.0)
    return out


# -- maths ---------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def normalize_to_u8(vol: np.ndarray, *, sample: int = 2_000_000) -> tuple[np.ndarray, tuple[float, float]]:
    """``(u8, (lo, hi))``: a float volume windowed to uint8 on a robust
    0.5-99.5 percentile range, and that range in data units (texture level
    ``k`` stands for the value ``lo + k / 255 * (hi - lo)``, which is what
    :func:`transfer_lut` needs to colour it as the slices do).

    The percentiles come from a strided sample: a 256-cubed volume has 16 M
    voxels and a full sort of them took a third of a second on the GUI thread.
    """
    v = np.ascontiguousarray(vol, dtype=np.float32)
    flat = v.reshape(-1)
    if flat.size > sample:
        flat = flat[:: flat.size // sample]
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.zeros(v.shape, dtype=np.uint8), (0.0, 1.0)
    lo, hi = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    out = (v - np.float32(lo)) * np.float32(255.0 / (hi - lo))
    np.clip(out, 0.0, 255.0, out=out)
    return np.nan_to_num(out).astype(np.uint8), (float(lo), float(hi))


def transfer_lut(display, tex_range: tuple[float, float],
                 window: tuple[float, float]) -> np.ndarray:
    """The colour of each of the 256 texture levels, (256, 4) uint8: the
    slice colouring (:func:`bidsmgr.viz.compute.intensity.colorize`: colour
    map, window, gamma, invert, negative tail, what is hidden below the
    window) applied to the value each level stands for. The 3-D render looks
    it up, so it is coloured as the slices are, by construction."""
    from .compute import intensity

    lo, hi = float(tex_range[0]), float(tex_range[1])
    values = np.linspace(lo, hi, 256, dtype=np.float32)[None, :]
    look = display.model_copy(update={"outline_px": 0.0})
    return np.ascontiguousarray(intensity.colorize(values, look, is_base=True,
                                                   window=window)[0])


def rgb_to_u8(vol: np.ndarray) -> np.ndarray:
    """Scale a colour volume by ONE window across channels (keeps the hue)."""
    rgb = np.asarray(vol[..., :3], dtype=np.float32)
    hi = float(np.nanmax(rgb)) if rgb.size else 0.0
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    scale = 255.0 if hi <= 1.0 + 1e-6 else 255.0 / hi
    return np.clip(np.nan_to_num(rgb) * scale, 0, 255).astype(np.uint8)


def clip_normal_from(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Unit clip-plane normal from azimuth/elevation (RAS texcoord space)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    n = _normalize(np.array([ce * np.sin(az), ce * np.cos(az), np.sin(el)], np.float32))
    return float(n[0]), float(n[1]), float(n[2])


def clip_uniforms(az: float, el: float, pos: float, thick: float, flip: bool):
    """(normal, depth, thickness) the shader uses for one clip plane."""
    n = clip_normal_from(az, el)
    if flip:
        n = (-n[0], -n[1], -n[2])
    depth = (pos - 0.5) * 1.8
    thick_u = 0.02 + thick * 2.98
    return n, float(depth), float(thick_u)


#: Clip planes the shader takes at once (``Scene.clips`` is capped the same).
MAX_CLIP_PLANES = 6


def clip_plane_uniforms(clips, cursor_tex=None) -> list[tuple[tuple[float, float, float], float, float]]:
    """``(normal, depth, thickness)`` of every ACTIVE clip plane, in order,
    at most :data:`MAX_CLIP_PLANES`.

    With ``cursor_tex`` (the crosshair as a texture coordinate) every plane
    is moved to pass through it, keeping its angle: the cut follows the
    crosshair as the slices do.
    """
    out = []
    for clip in clips or ():
        if not clip.active:
            continue
        normal, depth, thick = clip_uniforms(clip.az, clip.el, clip.pos, clip.thick, clip.flip)
        if cursor_tex is not None:
            depth = float(np.dot(np.asarray(normal), np.asarray(cursor_tex, dtype=float) - 0.5))
        out.append((normal, depth, thick))
        if len(out) >= MAX_CLIP_PLANES:
            break
    return out


def cut_mask(points: np.ndarray, planes, *, cut_away: bool = False) -> np.ndarray:
    """Which of ``points`` (N, 3 texture coordinates) the planes cut: any
    plane, or with ``cut_away`` only where every plane does (the shader's
    rule, for the pick)."""
    p = np.asarray(points, dtype=float) - 0.5
    if not planes:
        return np.zeros(len(p), dtype=bool)
    hits = []
    for normal, depth, thick in planes:
        sd = p @ np.asarray(normal, dtype=float)
        hits.append((sd > depth) & (sd < depth + thick))
    return np.logical_and.reduce(hits) if cut_away else np.logical_or.reduce(hits)


def ray_from_ndc(inv_view_proj: np.ndarray, nx: float, ny: float) -> tuple[np.ndarray, np.ndarray]:
    """Origin and unit direction of the ray through a pixel, in the render
    box's space (the shader's ``ro`` and ``rd``). ``nx, ny`` are normalised
    device coordinates (-1..1, y up)."""
    m = np.asarray(inv_view_proj, dtype=float)
    pn = m @ np.array([nx, ny, -1.0, 1.0])
    pf = m @ np.array([nx, ny, 1.0, 1.0])
    ro = pn[:3] / pn[3]
    rd = _normalize(pf[:3] / pf[3] - ro)
    return ro, rd


def intersect_box(ro: np.ndarray, rd: np.ndarray, box_half) -> Optional[tuple[float, float]]:
    """Where a ray enters and leaves the box ``[-box_half, box_half]``, or
    None when it misses."""
    half = np.asarray(box_half, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.asarray(rd, dtype=float)
    t0 = (-half - ro) * inv
    t1 = (half - ro) * inv
    near = float(np.max(np.minimum(t0, t1)))
    far = float(np.min(np.maximum(t0, t1)))
    if not np.isfinite(near) or not np.isfinite(far) or far < max(near, 0.0):
        return None
    return max(near, 0.0), far


def pick_depth(
    volume: np.ndarray, box_half, ro: np.ndarray, rd: np.ndarray, *,
    lo: float, hi: float, density: float, clips=(), steps: int = 512,
    surface: bool = True, cut_away: bool = False,
) -> Optional[np.ndarray]:
    """The texture coordinate (0..1 per axis) a click lands on.

    Marches the uploaded 8-bit volume along the ray the way the shader does
    and stops where the accumulated opacity passes one half: the first
    surface the eye sees, behind the clip planes. For the see-through
    effects (``surface=False``: MIP, X-ray) the densest sample wins instead.
    ``volume`` is (x, y, z) uint8, as uploaded; a colour volume's vector
    length is used. None when the ray misses the box or hits nothing.
    """
    hit = intersect_box(ro, rd, box_half)
    if hit is None:
        return None
    t_near, t_far = hit
    n = max(int(steps), 8)
    t = np.linspace(t_near, t_far, n)
    box = 2.0 * np.asarray(box_half, dtype=float)
    pts = ro[None, :] + rd[None, :] * t[:, None]
    p = (pts + np.asarray(box_half, dtype=float)) / box          # (n, 3) in 0..1
    keep = ~cut_mask(p, list(clips), cut_away=cut_away)
    dims = np.asarray(volume.shape[:3])
    idx = np.clip((p * dims).astype(int), 0, dims - 1)
    vals = volume[idx[:, 0], idx[:, 1], idx[:, 2]]
    if vals.ndim == 2:
        d = np.sqrt((vals.astype(np.float32) ** 2).sum(axis=1)) / 255.0
        d = np.clip(d, 0.0, 1.0)
    else:
        d = vals.astype(np.float32) / 255.0
    d[~keep] = 0.0
    e0, e1 = min(lo, hi), max(lo, hi)
    if e1 <= e0:
        e1 = e0 + 1e-3
    if not surface:
        if d.max() <= e0:
            return None
        return p[int(np.argmax(d))]
    x = np.clip((d - e0) / (e1 - e0), 0.0, 1.0)
    op = np.clip(x * x * (3.0 - 2.0 * x) * density, 0.0, 1.0)
    dt = (t_far - t_near) / n
    ref = float(np.linalg.norm(box)) / 512.0
    a = 1.0 - np.power(1.0 - op, dt / max(ref, 1e-9))
    acc = 1.0 - np.cumprod(1.0 - a)
    where = np.nonzero(acc >= 0.5)[0]
    if where.size == 0:
        return None
    return p[int(where[0])]


def texcoord_of_world(canon_inv_affine: np.ndarray, dims, world) -> np.ndarray:
    """A world point as the render texture's coordinate (0..1 per axis)."""
    w = np.asarray(world, dtype=float)
    c = np.asarray(canon_inv_affine, dtype=float)[:3, :3] @ w + np.asarray(canon_inv_affine)[:3, 3]
    return (c + 0.5) / np.asarray(dims, dtype=float)


def world_of_texcoord(canon_affine: np.ndarray, dims, texcoord) -> np.ndarray:
    """The inverse of :func:`texcoord_of_world`."""
    c = np.asarray(texcoord, dtype=float) * np.asarray(dims, dtype=float) - 0.5
    a = np.asarray(canon_affine, dtype=float)
    return a[:3, :3] @ c + a[:3, 3]


#: Shapes (MRS voxels) the ray caster intersects analytically, at most.
MAX_SHAPES = 4


def shape_uniforms(boxes, canon_affine: np.ndarray, dims, box_half) -> tuple[np.ndarray, np.ndarray]:
    """For each box (a :class:`~bidsmgr.viz.compute.shapes.GridBox`), the
    4 x 4 that takes the ray caster's box space (centred, half extents
    ``box_half``) into the box's own unit cube, and the box's edge lengths
    in box-space units. ``(MAX_SHAPES, 4, 4)`` and ``(MAX_SHAPES, 3)``,
    padded with identity and zeros.

    The chain: box space -> texture coordinate (0..1) -> canonical voxel ->
    world -> the shape's unit cube. One matrix per shape, so the shader
    tests a ray against it exactly instead of sampling a texture (a 20 mm
    voxel sampled at 20 mm a texel landed up to 10 mm off)."""
    half = np.asarray(box_half, dtype=float)
    d = np.asarray(dims, dtype=float)
    to_tex = np.eye(4)
    to_tex[:3, :3] = np.diag(1.0 / (2.0 * half))
    to_tex[:3, 3] = 0.5
    tex_to_vox = np.eye(4)
    tex_to_vox[:3, :3] = np.diag(d)
    tex_to_vox[:3, 3] = -0.5
    ray_to_world = np.asarray(canon_affine, dtype=float) @ tex_to_vox @ to_tex
    mats = np.tile(np.eye(4, dtype=np.float32), (MAX_SHAPES, 1, 1))
    lens = np.zeros((MAX_SHAPES, 3), dtype=np.float32)
    for i, box in enumerate(list(boxes)[:MAX_SHAPES]):
        m = box.unit_from_world() @ ray_to_world
        mats[i] = m
        lens[i] = np.linalg.norm(np.linalg.inv(m[:3, :3]), axis=0)
    return mats, lens


def light_dir_view(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Light direction in view space (x right, y up, z toward camera)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    d = _normalize(np.array([ce * np.sin(az), np.sin(el), ce * np.cos(az)], np.float32))
    return float(d[0]), float(d[1]), float(d[2])


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(r: float, t: float, n: float, f: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 / r
    m[1, 1] = 1.0 / t
    m[2, 2] = -2.0 / (f - n)
    m[2, 3] = -(f + n) / (f - n)
    return m


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
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


def camera_eye(az: float, el: float, dist: float, target) -> np.ndarray:
    ce = np.cos(el)
    d = np.array([ce * np.sin(az), ce * np.cos(az), np.sin(el)])
    return (np.asarray(target, dtype=float) + dist * d).astype(np.float32)


def camera_basis(az: float, el: float, dist: float, target):
    eye = camera_eye(az, el, dist, target)
    up = np.array([0, 0, 1], np.float32)
    f = _normalize(np.asarray(target, np.float32) - eye)
    r = _normalize(np.cross(f, up))
    return r, np.cross(r, f)


def make_matcap(name: str, size: int = 256) -> np.ndarray:
    """Procedural studio-lit sphere -> an RGB matcap (size x size x 3, uint8)."""
    amb, key_i, fill_i, spow, spec_i, tint = MATERIALS[name]
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


def cube_geometry() -> np.ndarray:
    """Interleaved [pos3, normal3, uv2] for a labelled orientation cube."""
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
        u0, v0 = col / 3.0, (1 - vrow) * 0.5

        def corner(iu, iv):
            pos = n + (2 * iu - 1) * right + (2 * iv - 1) * upv
            return [*pos, *nrm, u0 + iu / 3.0, v0 + iv * 0.5]

        c = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]
        for a, b, d in ((0, 1, 2), (0, 2, 3)):
            verts += [c[a], c[b], c[d]]
    return np.array(verts, np.float32)


__all__ = [
    "EFFECTS", "EFFECT_FX", "EFFECT_PARAMS", "EFFECT_PRESET", "LIGHTINGS",
    "MATERIALS", "MAX_CLIP_PLANES", "MAX_SHAPES", "PARAM_GROUPS", "PARAMS", "PARAM_BY_KEY", "Param",
    "QUALITY_DEFAULT", "QUALITY_MAX", "QUALITY_MIN", "SLICE_DEFAULT_ON",
    "SHARED_PARAMS", "THRESH_GAP", "baseline", "camera_basis", "camera_eye",
    "clamp_param", "clip_normal_from", "clip_plane_uniforms", "clip_uniforms",
    "cube_geometry", "cut_mask", "effective", "intersect_box", "light_dir_view", "look_at", "make_matcap",
    "normalize_to_u8", "ortho", "perspective", "pick_depth", "ray_from_ndc",
    "rgb_to_u8", "shape_uniforms", "texcoord_of_world", "transfer_lut", "uniform_values", "values_for",
    "world_of_texcoord",
]
