"""The 3-D canvas: a single-pass GPU ray-caster drawing the scene's volume.

The technique MRIcroGL uses, with only what BIDS Manager already ships
(PyQt6 + PyOpenGL), on a portable OpenGL 3.3 core profile. Everything the
render LOOKS like comes from the scene (``Scene.render``, ``Scene.clips``,
``Scene.display``) through :mod:`bidsmgr.viz.render3d`; this module uploads
textures and draws, and turns mouse gestures into commands.

What changed from the old widget, and why:

* The look is no longer ~25 public attributes poked by a control panel. The
  panel runs commands, the scene changes, this canvas reads it. Two linked
  viewers therefore share a render by sharing scene state, not widget names.
* The volume is normalised to 8 bits on a worker and uploaded ONCE per
  change. The old pane uploaded every load twice on the GUI thread, each
  time sorting every voxel for its percentiles.
* The canvas is still CREATED with its viewer, before the window is shown:
  adding the first QOpenGLWidget to a visible window makes Qt recreate the
  native window on Windows and Linux (the "GUI closes and reopens" bug).
  Creating it is cheap; nothing is uploaded until a 3-D view is on screen.

What the slices show, the render shows too. The scene's overlays are
coloured and blended by the same code the 2-D views use
(:mod:`bidsmgr.viz.compute.overlay3d`), on a worker, and uploaded as ONE
RGBA texture over the base's box; the crosshair is three lines through the
cursor; every active clip plane cuts; and a click (no drag) moves the
crosshair to the surface under the pointer, the march done again on the CPU
against the uploaded volume (:func:`bidsmgr.viz.render3d.pick_depth`).

GPU gate: :func:`gpu_available` reports whether an OpenGL 3.3 core context
on real hardware exists; layouts leave 3-D out when it does not.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ....viz import inputmap, render3d, views
from ....viz.compute import overlay3d
from ..bridge import connect_while_alive
from ..context import ViewerContext

log = logging.getLogger(__name__)

GL = None  # bound on first use (PyOpenGL is imported lazily)


def _gl_format() -> QSurfaceFormat:
    """A portable OpenGL 3.3 core surface format (does not touch the default)."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    return fmt


def request_gl_format() -> QSurfaceFormat:
    """Register the 3.3 core format as the app default (call once, pre-app)."""
    fmt = _gl_format()
    QSurfaceFormat.setDefaultFormat(fmt)
    return fmt


_GPU_CACHE: Optional[bool] = None


def gpu_available() -> bool:
    """True when an OpenGL 3.3-core context on real hardware is reachable.

    Creates a throwaway offscreen context, checks it reports >= 3.3 and
    rejects software rasterisers (llvmpipe, softpipe, Microsoft GDI). Cached.
    Needs a live ``QApplication``; returns False without one.
    """
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    _GPU_CACHE = False
    try:
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext
        from PyQt6.QtWidgets import QApplication

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
            from OpenGL import GL as _GL

            renderer = (_GL.glGetString(_GL.GL_RENDERER) or b"").decode(errors="ignore")
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


def make_cube_atlas(size: int = 96, flip=(1.0, 1.0, 1.0)) -> np.ndarray:
    """The six orientation letters in a 3x2 RGBA atlas (QPainter-drawn).

    ``flip`` swaps a letter pair so a mirrored render still reads upright:
    the cube itself is never reflected, that would mirror the glyphs.
    """
    letters = [["R", "A", "S"], ["L", "P", "I"]]
    for col, f in enumerate(flip):
        if f < 0:
            letters[0][col], letters[1][col] = letters[1][col], letters[0][col]
    img = QImage(size * 3, size * 2, QImage.Format.Format_RGBA8888)
    img.fill(QColor(38, 44, 54))
    p = QPainter(img)
    font = QFont()
    font.setPixelSize(int(size * 0.6))
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor(235, 240, 248))
    for row in range(2):
        for col in range(3):
            rect = img.rect().adjusted(col * size, row * size, 0, 0)
            rect.setWidth(size)
            rect.setHeight(size)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, letters[row][col])
    p.end()
    img = img.mirrored(False, True)  # GL textures are bottom-up
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(img.height(), img.width(), 4).copy()


# --------------------------------------------------------------------------
# Shaders (unchanged from the viewer that introduced them)
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
// The scene's overlays, already coloured and blended as the slices draw
// them, on the same box as uVol (so one set of texture coordinates serves
// both). Straight alpha.
uniform sampler3D uOverlay;
uniform int   uHasOverlay;
// The base layer's 2-D colouring of each of the 256 texture levels (colour
// map, window, gamma, what is hidden below the window), so the render is
// coloured as the slices are.
uniform sampler2D uLut;
uniform int   uUseLut;
// 0..1: how strongly the overlays (and the crosshair) are laid over the
// tissue in front of them. 0 is plain depth: inside the head they show only
// where it is cut open.
uniform float uSeeThrough;
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
// Up to six clip planes. A point is cut when ANY of them cuts it, or with
// uClipCutaway only when EVERY one does (a wedge or a corner cut out).
uniform int   uClipCount;
uniform int   uClipCutaway;
uniform vec3  uClipNormal[6];
uniform float uClipDepth[6];
uniform float uClipThick[6];
uniform int   uSliceOverlay;   // draw the cut face as an intensity slice
uniform float uSliceDepth;     // 0..1 -> how deep the slice integrates
// The crosshair: three lines through the cursor (a texture coordinate),
// intersected with each ray exactly rather than sampled by the march.
uniform int   uCrosshair;
uniform vec3  uCursor;
uniform vec3  uCrossColor;
uniform float uCrossWidth;     // box units, the line's full width
// Shapes (MRS voxels): boxes intersected exactly, each in its unit cube.
uniform int   uShapeCount;
uniform mat4  uShapeUnit[4];   // box space -> the shape's [0,1]^3
uniform vec3  uShapeLen[4];    // its edge lengths, in box units
uniform vec4  uShapeColor[4];
uniform float uShapeWidth;     // box units, an edge's full width

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
vec4 lutAt(float d) {
    return texture(uLut, vec2((clamp(d, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5));
}
// A colour's hue at full brightness (white for grey), so a shading model
// keeps its own light and dark and takes only the colour map's hue.
vec3 hueOf(vec3 c) {
    float m = max(max(c.r, c.g), c.b);
    return m > 0.0031 ? c / m : vec3(1.0);
}
bool cutBy(int i, vec3 p) {
    float sd = dot(uClipNormal[i], p - vec3(0.5));
    return sd > uClipDepth[i] && sd < uClipDepth[i] + uClipThick[i];
}
// Which clip plane cuts p (-1: none): the first that does, or, cutting
// away, 0 when every one does.
int clipIndex(vec3 p) {
    if (uClipCount == 0) return -1;
    if (uClipCutaway == 1) {
        for (int i = 0; i < 6; ++i) {
            if (i >= uClipCount) break;
            if (!cutBy(i, p)) return -1;
        }
        return 0;
    }
    for (int i = 0; i < 6; ++i) {
        if (i >= uClipCount) break;
        if (cutBy(i, p)) return i;
    }
    return -1;
}
bool clipped(vec3 p) { return clipIndex(p) >= 0; }
// The plane whose face a ray meets on leaving the cut at p. Cropping, it is
// the plane that cut the sample before; cutting away, the first plane that
// no longer cuts.
int facePlane(int prev, vec3 p) {
    if (uClipCutaway == 0) return prev;
    for (int i = 0; i < 6; ++i) {
        if (i >= uClipCount) break;
        if (!cutBy(i, p)) return i;
    }
    return prev;
}
vec3 grad(vec3 p) {
    vec3 e = 1.0 / uTexSize;
    return vec3(samp(p+vec3(e.x,0,0))-samp(p-vec3(e.x,0,0)),
                samp(p+vec3(0,e.y,0))-samp(p-vec3(0,e.y,0)),
                samp(p+vec3(0,0,e.z))-samp(p-vec3(0,0,e.z)));
}
void blendInto(inout vec4 acc, vec3 rgb, float a) {
    acc.rgb += (1.0 - acc.a) * rgb * a;
    acc.a   += (1.0 - acc.a) * a;
}
// The overlays at p, composited in depth order with the tissue (``acc``:
// hidden by what is in front of them, as anything is) AND into an
// accumulator of their own that nothing hides (``xacc``), which the end of
// the ray lays over the result at the see-through strength: a ghost of what
// is inside, never a coat of paint. ``shade`` lights the overlay in ``acc``
// by the tissue's normal, so an atlas on the cortex reads as a coloured
// surface rather than a flat decal.
void addOverlay(inout vec4 acc, inout vec4 xacc, vec3 p, float stepRatio, vec3 shade) {
    vec4 o = texture(uOverlay, p);
    if (o.a > 0.004) {
        float a = 1.0 - pow(1.0 - clamp(o.a, 0.0, 1.0), stepRatio);
        blendInto(acc, o.rgb * shade, a);
        blendInto(xacc, o.rgb, a);
    }
}
// Where the ray meets the crosshair: the coverage of the nearest of its
// three lines (0: missed) and, in tHit, how far along the ray. Closest
// approach of two lines, the crosshair's clamped to the box.
float crossHit(vec3 ro, vec3 rd, out float tHit) {
    tHit = 1e9;
    if (uCrosshair == 0) return 0.0;
    vec3 c = uCursor * (2.0 * uBoxHalf) - uBoxHalf;
    vec3 w0 = ro - c;
    float cover = 0.0;
    for (int k = 0; k < 3; ++k) {
        vec3 e = vec3(0.0);
        e[k] = 1.0;
        float b = dot(rd, e);
        float denom = 1.0 - b * b;
        if (denom < 1e-6) continue;          // seen end-on: a point, not a line
        float d = dot(rd, w0);
        float f = dot(e, w0);
        float s = clamp((f - b * d) / denom, -uBoxHalf[k] - c[k], uBoxHalf[k] - c[k]);
        vec3 q = c + s * e;
        float t = dot(q - ro, rd);
        if (t < 0.0) continue;
        float dist = length(ro + rd * t - q);
        float k2 = 1.0 - smoothstep(uCrossWidth * 0.25, uCrossWidth * 0.5, dist);
        if (k2 > 0.01 && t < tHit) { tHit = t; cover = k2; }
    }
    return cover;
}
// How much of an edge of shape i covers the point q of its unit cube: an
// edge is where TWO coordinates are at a face, so the second smallest
// distance to a face (in box units) decides.
float shapeEdge(int i, vec3 q) {
    vec3 dist = min(q, 1.0 - q) * uShapeLen[i];
    float mn = min(dist.x, min(dist.y, dist.z));
    float mx = max(dist.x, max(dist.y, dist.z));
    float mid = dist.x + dist.y + dist.z - mn - mx;
    return 1.0 - smoothstep(uShapeWidth * 0.30, uShapeWidth * 0.55, mid);
}
// The nearest shape the ray meets, exactly (a slab test in its unit cube):
// its colour, and in .a its coverage (crisp edges in front, the edges behind
// at half strength, a faint face); in tHit, how far along the ray.
vec4 shapeHit(vec3 ro, vec3 rd, out float tHit) {
    tHit = 1e9;
    vec4 best = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        if (i >= uShapeCount) break;
        vec3 o = (uShapeUnit[i] * vec4(ro, 1.0)).xyz;
        vec3 d = mat3(uShapeUnit[i]) * rd;
        vec3 inv = vec3(abs(d.x) > 1e-12 ? 1.0 / d.x : 1e12,
                        abs(d.y) > 1e-12 ? 1.0 / d.y : 1e12,
                        abs(d.z) > 1e-12 ? 1.0 / d.z : 1e12);
        vec3 ta = -o * inv;
        vec3 tb = (vec3(1.0) - o) * inv;
        vec3 tmin = min(ta, tb), tmax = max(ta, tb);
        float tn = max(max(tmin.x, tmin.y), tmin.z);
        float tf = min(min(tmax.x, tmax.y), tmax.z);
        if (tf < max(tn, 0.0)) continue;
        float front = tn >= 0.0 ? shapeEdge(i, o + d * tn) : 0.0;
        float back = shapeEdge(i, o + d * tf);
        float a = max(max(front, back * 0.55), tn >= 0.0 ? 0.20 : 0.10) * uShapeColor[i].a;
        float tEnter = max(tn, 0.0);
        if (tEnter < tHit) { tHit = tEnter; best = vec4(uShapeColor[i].rgb, a); }
    }
    return best;
}
// A shape over the colour, faded by the tissue in front of it like the
// crosshair: never under half strength, so a voxel inside the head is found.
vec3 shapeOver(vec3 col, vec4 shape, float hidden) {
    if (shape.a <= 0.0) return col;
    float shown = 1.0 - clamp(hidden, 0.0, 1.0) * (1.0 - max(uSeeThrough, 0.5));
    return mix(col, shape.rgb, shape.a * shown);
}
// The ray's colour with the overlay ghost, the shapes and the crosshair laid
// over it. ``hidden`` is how much tissue lay in front of the crosshair: the
// cursor stays findable inside the head, never under half strength.
vec3 finish(vec3 col, vec4 xacc, float cover, float hidden, vec4 shape, float shapeHidden) {
    col = mix(col, xacc.rgb + (1.0 - xacc.a) * col, uSeeThrough);
    col = shapeOver(col, shape, shapeHidden);
    float shown = 1.0 - clamp(hidden, 0.0, 1.0) * (1.0 - max(uSeeThrough, 0.5));
    return mix(col, uCrossColor, cover * shown);
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
    float stepRatio = dt / refStep;
    float t0 = tN + hash(gl_FragCoord.xy) * dt;
    vec3 duvw = rd * dt / boxSize;

    float e0 = min(uThreshLo, uThreshHi);
    float e1 = max(uThreshLo, uThreshHi);
    if (e1 <= e0) e1 = e0 + 1e-3;

    bool overlays = uHasOverlay == 1;
    vec4 xacc = vec4(0.0);           // the overlays, unhidden
    vec4 none = vec4(0.0);           // MIP and X-ray hide nothing: no depth pass
    float crossT;
    float cover = crossHit(ro, rd, crossT);
    // How much tissue is in front of the crosshair, taken as the march
    // passes it. A little early, so a line lying ON a cut face (the cut
    // through the crosshair) is never hidden by that face.
    float crossFront = -1.0;
    float crossAt = crossT - 2.0 * dt;
    float shapeT;
    vec4 shape = shapeHit(ro, rd, shapeT);
    float shapeFront = -1.0;
    float shapeAt = shapeT - dt;

    // ---- MIP ----
    if (uEffect == 4) {
        float mx = 0.0, t = t0; vec3 tint = vec3(1.0);
        for (int i=0;i<4096;++i){ if(t>tF)break; vec3 p=(ro+rd*t+uBoxHalf)/boxSize;
            // Carry the hue of the brightest voxel so a colour-FA MIP keeps
            // its direction colouring instead of collapsing to grey.
            if(!clipped(p)){ float s=samp(p); if(s>mx){ mx=s; tint=voxelTint(p);}
                if (overlays) addOverlay(none, xacc, p, stepRatio, vec3(1.0)); } t+=dt; }
        float w = clamp((mx-e0)/(e1-e0),0.0,1.0);
        if (uUseLut == 1) { vec4 L = lutAt(mx); tint = L.rgb; w *= step(0.004, L.a); }
        vec3 col = mix(uBg, tint, w);
        // A projection hides nothing, so its overlays and crosshair are whole.
        col = xacc.rgb + (1.0 - xacc.a) * col;
        col = shapeOver(col, shape, 0.0);
        FragColor = vec4(mix(col, uCrossColor, cover), 1.0); return;
    }
    // ---- X-ray ----
    if (uEffect == 3) {
        float sum=0.0, t=t0; vec3 csum = vec3(0.0);
        for (int i=0;i<4096;++i){ if(t>tF)break; vec3 p=(ro+rd*t+uBoxHalf)/boxSize;
            if(!clipped(p)){ float d=samp(p);
                if(d>e0){ float w=(d-e0)*uDensity; sum+=w;
                    csum += (uUseLut == 1 ? lutAt(d).rgb : voxelTint(p)) * w; }
                if (overlays) addOverlay(none, xacc, p, stepRatio, vec3(1.0)); }
            t+=dt; }
        float a = 1.0 - exp(-sum*dt*6.0);
        // Density-weighted mean hue along the ray (white for scalar volumes).
        vec3 tint = sum > 1e-6 ? csum / sum : vec3(1.0);
        vec3 col = mix(uBg, tint, clamp(a,0.0,1.0));
        col = xacc.rgb + (1.0 - xacc.a) * col;
        col = shapeOver(col, shape, 0.0);
        FragColor = vec4(mix(col, uCrossColor, cover), 1.0); return;
    }

    vec3 viewDir = vec3(0.0, 0.0, 1.0);

    // ---- Opacity peeling (Rezk-Salama & Kolb): render the (peel+1)-th layer.
    //      Peeling (6) resets the accumulator on each layer; Peeling 2 (7)
    //      keeps prior layers faintly (translucent nested peel).
    if (uEffect==6 || uEffect==7) {
        vec4 acc = vec4(0.0); float pNum = 0.0; float t = t0;
        for (int i=0;i<4096;++i){
            if (t>tF) break;
            if (crossFront < 0.0 && t >= crossAt) crossFront = acc.a;
            if (shapeFront < 0.0 && t >= shapeAt) shapeFront = acc.a;
            vec3 p = (ro+rd*t+uBoxHalf)/boxSize;
            if (clipped(p)) { t+=dt; continue; }
            float d = samp(p);
            float a = smoothstep(e0,e1,d) * uDensity;
            if (uUseLut == 1) a *= step(0.004, lutAt(d).a);
            a = 1.0 - pow(1.0-clamp(a,0.0,1.0), stepRatio);
            if (overlays) addOverlay(acc, xacc, p, stepRatio, vec3(1.0));
            if (a > 0.01) {
                vec3 nv = normalize(uNormalMatrix * normalize(-grad(p)+1e-6));
                float ndl = max(dot(nv, uLightDir), 0.0);
                float sp = pow(max(dot(reflect(-uLightDir,nv),viewDir),0.0), max(uShininess,1.0))*uSpecular;
                vec3 base = uUseLut == 1 ? lutAt(d).rgb : vec3(pow(d,0.8));
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
        if (crossFront < 0.0) crossFront = acc.a;
        if (shapeFront < 0.0) shapeFront = acc.a;
        FragColor = vec4(finish(acc.rgb + (1.0-acc.a)*uBg, xacc, cover, crossFront,
                                shape, shapeFront), 1.0); return;
    }

    bool translucent = (uEffect==2 || uEffect==5 || uEffect==8);   // Glass/Edges/Shell
    vec3 edgeCol = tempTint(uColorTemp);
    // Once the tissue is opaque the march may stop, unless a ghost of the
    // overlays behind it is still being gathered.
    bool ghosts = overlays && uSeeThrough > 0.001;

    vec4 acc = vec4(0.0);
    float t = t0;
    int prevClip = -1;
    for (int i=0;i<4096;++i){
        if (t>tF) break;
        if (crossFront < 0.0 && t >= crossAt) crossFront = acc.a;
        if (shapeFront < 0.0 && t >= shapeAt) shapeFront = acc.a;
        if (acc.a>0.985 && !translucent && !(ghosts && xacc.a < 0.985)) break;
        vec3 p = (ro+rd*t+uBoxHalf)/boxSize;
        int ci = clipIndex(p);
        if (ci >= 0) { prevClip = ci; t += dt; continue; }
        float d = samp(p);

        // Cut face -> hybrid cross-section. SOLID tissue at the plane is drawn
        // as a clean flat intensity slice; low-intensity CSF / air is treated
        // as EMPTY (transparent) so the lit 3-D render behind shows through —
        // a ventricle becomes a real, shaded recess (that is the "depth"). The
        // overlay-depth slider raises the emptiness threshold; smoothstep keeps
        // the boundary smooth so there are no black-dot speckles. We do NOT
        // break: the loop continues and the normal surface shading below fills
        // the transparent parts. Scoped entirely to this block.
        if (uSliceOverlay==1 && prevClip >= 0) {
            int fi = facePlane(prevClip, p);
            vec3 cn = uClipNormal[fi];
            float cd = uClipDepth[fi], ct = uClipThick[fi];
            prevClip = -1;
            // De-jitter: snap to the exact clip-plane crossing for a clean value.
            vec3 q = p;
            float denom = dot(cn, duvw);
            if (abs(denom) > 1e-6) {
                float sdp = dot(cn, p - vec3(0.5));
                float tgt = (abs(sdp - cd) <= abs(sdp - (cd+ct))) ? cd : (cd + ct);
                q = clamp(p + duvw * ((tgt - sdp) / denom), vec3(0.0), vec3(1.0));
            }
            // The overlays lie ON the cut face, as they lie on a slice: drawn
            // at their full strength, over the face.
            if (overlays) addOverlay(acc, xacc, q, 1.0, vec3(1.0));
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
                if (uUseLut == 1) {
                    // The cut face IS the slice: the 2-D colouring exactly.
                    vec4 L = lutAt(sd);
                    face = (uEffect==10) ? brainTissue(sv) * hueOf(L.rgb) : L.rgb;
                    solid *= step(0.004, L.a);
                }
                acc.rgb += (1.0-acc.a) * face * solid;
                acc.a   += (1.0-acc.a) * solid;
            }
            t += dt;
            continue;   // let the lit 3-D volume fill the empty (cavity) parts
        }
        prevClip = -1;

        vec3 g = grad(p);
        float gm = length(g);
        vec3 nv = normalize(uNormalMatrix * normalize(-g + 1e-6));
        float op = smoothstep(e0, e1, d) * uDensity;
        float depth01 = clamp((t-tN)/max(tF-tN,1e-3), 0.0, 1.0);
        float ndl = max(dot(nv, uLightDir), 0.0);
        float sp = pow(max(dot(reflect(-uLightDir, nv), viewDir), 0.0), max(uShininess,1.0));

        // The overlays at p, lit by the tissue's own normal so an atlas on
        // the cortex reads as a coloured surface rather than a flat decal.
        if (overlays) {
            vec3 shade = vec3(uAmbient + uDiffuse * ndl) + vec3(sp * uSpecular * 0.5);
            addOverlay(acc, xacc, p, stepRatio, clamp(shade, 0.35, 1.6));
        }

        // The 2-D colour map's colour at this intensity (and whether the
        // slices hide it): the base of every surface effect when on.
        vec4 L = uUseLut == 1 ? lutAt(d) : vec4(1.0);
        if (uUseLut == 1) op *= step(0.004, L.a);
        vec3 hue = uUseLut == 1 ? hueOf(L.rgb) : vec3(1.0);
        vec3 lit; float a = op;
        if (uEffect==0) {                                  // fx0 = "Matte" label (waxy matcap)
            vec3 mc = texture(uMatcap, nv.xy*0.5+0.5).rgb;
            vec3 surf = mix(vec3(0.74) * hue, uUseLut == 1 ? L.rgb : vec3(pow(d,0.72)), uSurface);
            // Waxy matcap base lit with the SAME Phong model as Matte
            // (ambient + diffuse*N·L, plus specular/shininess), so the lighting
            // controls behave identically between Default and Matte.
            vec3 base = mc * surf * uBrighten;
            lit = base * (uAmbient + uDiffuse * ndl) + vec3(sp * uSpecular);
            lit *= 1.0 - 0.18 * depth01;
            a = op * 0.92;
        } else if (uEffect==9) {                           // Topography
            vec3 mc = texture(uMatcap, nv.xy*0.5+0.5).rgb;
            vec3 surf = mix(vec3(0.74) * hue, uUseLut == 1 ? L.rgb : vec3(pow(d,0.72)), uSurface);
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
            vec3 surf = mix(vec3(0.6) * hue, uUseLut == 1 ? L.rgb : vec3(pow(d,0.8)), uSurface);
            lit = mc * surf * uBrighten * (0.35 + 0.9*edge);
            a = op * mix(0.02, 0.9, mix(bound, edge, uEdgeMix));
        } else if (uEffect==2 || uEffect==8) {             // Glass / Shell
            float edge = smoothstep(uEdgeThresh, 1.0, gm);
            float bnd = (uEffect==2) ? ((gm>uBoundThresh)?pow(1.0-abs(nv.z),4.0):0.0)
                                     : smoothstep(uBoundThresh, 1.0, gm);
            float e = mix(bnd, edge, uEdgeMix);
            lit = edgeCol * hue * (e * uBrighten + sp * uSpecular);
            a = e * (uEffect==2 ? 0.35 : 0.7);
        } else if (uEffect==10) {                          // Realistic (living tissue)
            // Same Phong response as Standard, but the base is tissue-coloured
            // and gains two touches that sell "alive":
            //  * a rim of subsurface red where the surface turns away from the
            //    viewer, mimicking light scattering through thin tissue;
            //  * a WHITE specular, because a wet surface reflects the light's
            //    colour, not the tissue's — tinting it pink looks like plastic.
            vec3 base = brainTissue(pow(d, 0.85)) * hue;
            // Subsurface warmth only at grazing angles, and gently: at the
            // 0.45 it started out it tinted the whole surface salmon.
            float rim = pow(1.0 - abs(nv.z), 4.0);
            base = mix(base, base * vec3(1.14, 0.80, 0.75), 0.20 * rim);
            lit = base * (uAmbient + uDiffuse*ndl) + vec3(1.0, 0.97, 0.94)*(sp*uSpecular);
            lit *= 1.0 - 0.18 * depth01;
            a = op;
        } else {                                           // fx1 = "Standard" label (flat Phong)
            vec3 base = uUseLut == 1 ? mix(vec3(0.12) * hue, L.rgb, 0.88)
                                     : vec3(mix(0.5, pow(d,0.8), 0.7));
            lit = base * (uAmbient + uDiffuse*ndl) + vec3(sp*uSpecular);
            lit *= 1.0 - 0.30 * depth01;
            a = op;
        }
        lit *= voxelTint(p);          // white for scalar data, hue for colour-FA
        if (a > 0.0008) {
            a = 1.0 - pow(1.0 - clamp(a,0.0,1.0), stepRatio);
            acc.rgb += (1.0-acc.a) * lit * a;
            acc.a   += (1.0-acc.a) * a;
        }
        t += dt;
    }
    if (crossFront < 0.0) crossFront = acc.a;
    if (shapeFront < 0.0) shapeFront = acc.a;
    FragColor = vec4(finish(acc.rgb + (1.0-acc.a)*uBg, xacc, cover, crossFront,
                            shape, shapeFront), 1.0);
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


def canonical_volume(frame: np.ndarray, src) -> tuple[np.ndarray, tuple[float, float, float]]:
    """``frame`` turned to RAS order (x, y, z) for the texture, with spacing.

    The texture's axes ARE the render box's axes, so they must be the
    anatomical ones whatever order the file stores; this is the same
    reorientation the 2-D views do per plane, done once for the volume.
    """
    ornt = views.orientation(src)
    order = list(ornt.data_of)          # data axis for RAS x, y, z
    vol = np.transpose(frame, order + list(range(3, frame.ndim)))
    for ras in range(3):
        if ornt.sign[ornt.data_of[ras]] < 0:
            vol = np.flip(vol, axis=ras)
    spacing = tuple(float(src.zooms3[d]) for d in order)
    return vol, spacing  # type: ignore[return-value]


def prepare_upload(frame: np.ndarray, src, is_rgb: bool):
    """Worker-side: reorient and quantise a frame for the GPU. Returns
    ``(u8, spacing, (lo, hi))``, the range in data units the 8 bits span."""
    vol, spacing = canonical_volume(frame, src)
    if is_rgb:
        u8, rng = render3d.rgb_to_u8(vol), (0.0, 1.0)
    else:
        u8, rng = render3d.normalize_to_u8(src.scale(vol))
    return np.ascontiguousarray(u8), spacing, rng


def canonical_geometry(src) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    """The affine, dims and spacing of the base's RAS-ordered copy: what the
    render texture's coordinates mean in the world."""
    ornt = views.orientation(src)
    order = list(ornt.data_of)
    dims = tuple(int(src.spatial[d]) for d in order)
    spacing = tuple(float(src.zooms3[d]) for d in order)
    affine = overlay3d.canonical_affine(src.affine, src.spatial, ornt)
    return affine, dims, spacing  # type: ignore[return-value]


def prepare_overlays(canon_affine, base_dims, base_spacing, inputs, cache, *, cancel=None):
    """Worker-side: the overlays blended into one RGBA volume (see
    :func:`overlay3d.build_overlay_volume`). ``inputs`` carry raw frames;
    scaling to data units happens here, after the resample, on the smaller
    array."""
    rgba, dims, kept = overlay3d.build_overlay_volume(
        canon_affine, base_dims, base_spacing, inputs, cache=cache, cancel=cancel)
    if rgba is None:
        return None
    smooth = any(ov.display.interpolation != "nearest" and ov.display.label_table is None
                 for ov in inputs)
    return np.ascontiguousarray(rgba.transpose(2, 1, 0, 3)), dims, kept, smooth


class RenderCanvas(QOpenGLWidget):
    """The scene's base volume, ray-cast. Reads the scene; runs commands."""

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setFormat(_gl_format())
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumSize(60, 60)
        self._prog = 0
        self._vao = 0
        self._tex = 0
        self._matcap_tex = 0
        self._matcap_name = ""
        self._cube_prog = 0
        self._cube_vao = 0
        self._cube_vbo = 0
        self._cube_tex = 0
        self._cube_nverts = 0
        self._cube_flip: tuple = (1.0, 1.0, 1.0)
        self._tex_dims = (1, 1, 1)
        self._box_half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._gl_ok = False
        self.gl_error = ""
        # The prepared volume waiting for (or already in) the texture.
        self._pending: Optional[tuple[np.ndarray, tuple]] = None
        self._uploaded_key: Optional[tuple] = None
        self._wanted_key: Optional[tuple] = None
        self._is_rgb = False
        #: World geometry of the uploaded (or wanted) base: canonical affine,
        #: dims and spacing. The crosshair and the pick convert through it.
        self._geom: Optional[tuple] = None
        self._wanted_geom: Optional[tuple] = None
        # The overlays, as one RGBA texture over the same box.
        self._ov_tex = 0
        #: A 1-voxel transparent volume bound when there is no overlay: a
        #: sampler with nothing bound is undefined (macOS logs it, others may
        #: draw garbage).
        self._ov_empty = 0
        # The transfer table: the base layer's 2-D colouring of every texture
        # level, a 256 x 1 texture rebuilt when the look changes (1 KB).
        self._lut_tex = 0
        self._lut_key: Optional[tuple] = None
        self._tex_range = (0.0, 1.0)
        self._ov_pending: Optional[tuple] = None
        self._ov_key: Optional[tuple] = None
        self._ov_wanted: Optional[tuple] = None
        self._ov_cache: dict = {}
        self._ov_smooth = True
        self._last: Optional[QPoint] = None
        self._press: Optional[QPoint] = None
        self._moved = False
        self._tool = "none"
        ctx.qstore.changed.connect(self._on_changed)
        ctx.jobs.done.connect(self._on_job_done)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.update())

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def gl_ok(self) -> bool:
        return self._gl_ok

    def has_volume(self) -> bool:
        return self._uploaded_key is not None

    def _volume_key(self) -> Optional[tuple]:
        layer, src = views.base(self.ctx.store)
        if layer is None or src is None:
            return None
        t = views.frame_of(self.ctx.store, layer, src)
        if not src.frame_ready(t):
            return None
        return (layer.source, t, id(src))

    def has_overlay(self) -> bool:
        """Whether an overlay volume has been prepared for the render (and
        uploaded, where there is a GL context)."""
        return bool(self._ov_key and self._ov_key[1]) and self._ov_pending is not None

    def _on_changed(self, paths) -> None:
        repaint = False
        for p in paths:
            if p.startswith(("views:", "graph")):
                continue
            if p == "cursor":
                # The crosshair moved: redraw (it is drawn, or the cut follows
                # it), nothing to re-upload.
                repaint = repaint or (self.ctx.scene.display.crosshair
                                      or self.ctx.scene.render.cut_at_cursor)
                continue
            self.refresh_volume()
            self.refresh_overlays()
            self.update()
            return
        if repaint:
            self.update()

    def refresh_volume(self) -> None:
        """Prepare the base frame for the GPU on a worker, if it changed.

        Only while the canvas is on screen: a hidden 3-D view costs nothing.
        """
        if not self.isVisible():
            return
        key = self._volume_key()
        if key is None or key == self._uploaded_key or key == self._wanted_key:
            return
        layer, src = views.base(self.ctx.store)
        raw = src.raw_frame(key[1])
        if raw is None or not (raw.ndim == 3 or (src.is_rgb and raw.ndim == 4)):
            return
        self._wanted_key = key
        self._wanted_geom = canonical_geometry(src)
        self._is_rgb = bool(src.is_rgb)
        self.ctx.jobs.start("render-volume", hash(key) & 0x7FFFFFFF,
                            prepare_upload, np.asarray(raw), src, src.is_rgb)

    # -- overlays -------------------------------------------------------------

    def _overlay_key(self) -> Optional[tuple]:
        """What the overlay texture depends on: the base's grid and every
        overlay drawn in 3-D with its frame and look. None without a base."""
        store = self.ctx.store
        base_layer, base_src = views.base(store)
        if base_layer is None or base_src is None:
            return None
        values = render3d.values_for(self.ctx.scene.render)
        if values.get("layers", 1.0) < 0.5:
            return ((base_layer.source, id(base_src)), ())
        parts = []
        for layer in self.ctx.scene.layers:
            if (layer.kind != "volume" or layer.id == base_layer.id
                    or not layer.visible or not layer.in_3d):
                continue
            src = views.source_of(store, layer)
            if src is None or views.is_shape(src):
                continue
            t = views.frame_of(store, layer, src)
            parts.append((layer.id, layer.source, id(src), t, src.frame_ready(t),
                          layer.display.model_dump_json()))
        return ((base_layer.source, id(base_src)), tuple(parts))

    def _overlay_inputs(self) -> list:
        store = self.ctx.store
        base_layer, _base_src = views.base(store)
        out = []
        for layer in self.ctx.scene.layers:
            if (layer.kind != "volume" or base_layer is None or layer.id == base_layer.id
                    or not layer.visible or not layer.in_3d):
                continue
            src = views.source_of(store, layer)
            if src is None or views.is_shape(src):
                continue
            t = views.frame_of(store, layer, src)
            raw = src.raw_frame(t)
            if raw is None:
                continue
            # An atlas is coloured by its table; only a scalar map needs a window.
            window = layer.display.window
            if window is None:
                window = ((0.0, 1.0) if layer.display.label_table is not None
                          else src.robust_range(t) or (0.0, 1.0))
            out.append(overlay3d.OverlayInput(
                key=(layer.source, id(src), t), values=np.asarray(raw),
                inv_affine=np.asarray(src.inv_affine), zooms=tuple(src.zooms3),
                display=layer.display.model_copy(), window=tuple(window),
                slope=float(src.slope), inter=float(src.inter),
            ))
        return out

    def refresh_overlays(self) -> None:
        """Rebuild the overlay texture on a worker when an overlay, its look,
        its frame or the base changed. Nothing runs while nothing changed."""
        if not self.isVisible():
            return
        key = self._overlay_key()
        if key == self._ov_key or key == self._ov_wanted:
            return
        if key is None or not key[1]:
            self.ctx.jobs.cancel("render-overlays")
            self._ov_wanted = None
            self._ov_key = key
            self._ov_cache = {}
            self._drop_overlay_texture()
            self.update()
            return
        _layer, src = views.base(self.ctx.store)
        affine, dims, spacing = canonical_geometry(src)
        inputs = self._overlay_inputs()
        self._ov_wanted = key
        self.ctx.jobs.start("render-overlays", hash(key) & 0x7FFFFFFF, prepare_overlays,
                            affine, dims, spacing, inputs, self._ov_cache)

    def _drop_overlay_texture(self) -> None:
        self._ov_pending = None
        if self._ov_tex and self._make_current():
            try:
                GL.glDeleteTextures([self._ov_tex])
            except Exception:  # noqa: BLE001
                pass
            finally:
                self.doneCurrent()
        self._ov_tex = 0

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "render-overlays":
            if self._ov_wanted is None or (hash(self._ov_wanted) & 0x7FFFFFFF) != generation:
                return
            key, self._ov_wanted = self._ov_wanted, None
            if result is None:
                self._ov_key = key
                self._drop_overlay_texture()
                self.update()
                return
            data, _dims, cache, smooth = result
            self._ov_cache = cache
            self._ov_smooth = bool(smooth)
            self._ov_pending = data
            if self._gl_ok and self._make_current():
                try:
                    self._upload_overlay()
                finally:
                    self.doneCurrent()
            self._ov_key = key
            self.update()
            return
        if tag != "render-volume" or self._wanted_key is None:
            return
        if (hash(self._wanted_key) & 0x7FFFFFFF) != generation:
            return
        self._pending = result
        key, self._wanted_key = self._wanted_key, None
        self._geom, self._wanted_geom = self._wanted_geom, None
        if self._gl_ok and self._make_current():
            try:
                self._upload()
                self._uploaded_key = key
            finally:
                self.doneCurrent()
        else:
            self._uploaded_key = key
        self.update()
        # A new base means a new box: the overlays are resampled onto it.
        self.refresh_overlays()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh_volume()
        self.refresh_overlays()

    def clear(self) -> None:
        self._pending = None
        self._uploaded_key = None
        self._wanted_key = None
        self._geom = None
        self._wanted_geom = None
        self._ov_key = None
        self._ov_wanted = None
        self._ov_cache = {}
        self.ctx.jobs.cancel("render-overlays")
        self._drop_overlay_texture()
        if self._tex and self._make_current():
            try:
                GL.glDeleteTextures([self._tex])
                self._tex = 0
            except Exception:  # noqa: BLE001
                pass
            finally:
                self.doneCurrent()
        self.update()

    # ------------------------------------------------------------------
    # Flip: the 3-D render mirrors with the 2-D views
    # ------------------------------------------------------------------

    def display_flip(self) -> tuple[float, float, float]:
        """Per RAS axis display mirror, matching the 2-D views.

        A look-at view of the volume is the mirror image of a slice shown in
        the neurological convention, hence the constant extra L/R mirror.
        """
        d = self.ctx.scene.display
        flip = [1.0, 1.0, 1.0]
        _layer, src = views.base(self.ctx.store)
        if not d.ras and src is not None:
            ornt = views.orientation(src)
            for ras in range(3):
                if ornt.sign[ornt.data_of[ras]] < 0:
                    flip[ras] = -1.0
        if d.radiological:
            flip[0] = -flip[0]
        flip[0] = -flip[0]
        return flip[0], flip[1], flip[2]

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def _make_current(self) -> bool:
        try:
            self.makeCurrent()
            return self.context() is not None and self.context().isValid()
        except Exception:  # noqa: BLE001
            return False

    def initializeGL(self) -> None:  # noqa: N802
        global GL
        self._gl_ok = False
        self._tex = 0
        try:
            from OpenGL import GL as _GL

            GL = _GL
            self._prog = self._build_program(_VERT, _FRAG)
            self._vao = GL.glGenVertexArrays(1)
            self._matcap_tex = GL.glGenTextures(1)
            self._matcap_name = ""
            self._ov_tex = 0
            self._ov_empty = self._make_empty_volume()
            self._lut_tex = GL.glGenTextures(1)
            self._lut_key = None
            self._init_cube()
            GL.glDisable(GL.GL_DEPTH_TEST)
            self._gl_ok = True
            if self._pending is not None:
                self._upload()
            if self._ov_pending is not None:
                self._upload_overlay()
        except Exception as exc:  # noqa: BLE001
            self._gl_ok = False
            self.gl_error = str(exc)
            log.warning("3-D GL initialisation failed: %s", exc)

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        if self._gl_ok:
            GL.glViewport(0, 0, w, h)

    def _upload(self) -> None:
        if self._pending is None:
            return
        u8, spacing, self._tex_range = self._pending
        rgb = u8.ndim == 4
        x, y, z = u8.shape[:3]
        self._tex_dims = (x, y, z)
        extent = np.array([x * spacing[0], y * spacing[1], z * spacing[2]], np.float32)
        self._box_half = (0.5 * extent / float(max(extent.max(), 1e-6))).astype(np.float32)
        data = np.ascontiguousarray(u8.transpose(2, 1, 0, 3) if rgb else u8.transpose(2, 1, 0))
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
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data[..., :3].copy() if data.shape[-1] > 3 else data)
        else:
            GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R8, x, y, z, 0,
                            GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)

    def _lut(self) -> Optional[np.ndarray]:
        """The base layer's transfer table, or None (an RGB volume, the
        option off, nothing open)."""
        rs = self.ctx.scene.render
        if not rs.use_colormap or self._is_rgb:
            return None
        layer, src = views.base(self.ctx.store)
        if layer is None or src is None:
            return None
        window = layer.display.window
        if window is None:
            window = src.robust_range(views.frame_of(self.ctx.store, layer, src)) or (0.0, 1.0)
        key = (layer.display.model_dump_json(), tuple(window), self._tex_range)
        if key != self._lut_key:
            self._lut_table = render3d.transfer_lut(layer.display, self._tex_range, window)
            self._lut_key = key
            self._lut_dirty = True
        return self._lut_table

    def _bind_lut(self, unit: int) -> bool:
        """Upload the table if it changed and bind it; False when unused."""
        table = self._lut()
        if table is None or not self._lut_tex:
            return False
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._lut_tex)
        if getattr(self, "_lut_dirty", True):
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            for pn in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, pn, GL.GL_CLAMP_TO_EDGE)
            # Nearest: a label-like or stepped colour map keeps its steps.
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 256, 1, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(table))
            self._lut_dirty = False
        return True

    def _make_empty_volume(self) -> int:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA8, 1, 1, 1, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.zeros(4, np.uint8))
        return tex

    def _upload_overlay(self) -> None:
        if self._ov_pending is None:
            return
        data = self._ov_pending          # (z, y, x, 4) uint8, C order
        z, y, x = data.shape[:3]
        if self._ov_tex:
            GL.glDeleteTextures([self._ov_tex])
        self._ov_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._ov_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        for pn in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R):
            GL.glTexParameteri(GL.GL_TEXTURE_3D, pn, GL.GL_CLAMP_TO_EDGE)
        # An atlas or a mask is blocky by nature: linear filtering would blend
        # two regions' colours into a third along every border.
        filt = GL.GL_LINEAR if self._ov_smooth else GL.GL_NEAREST
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, filt)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, filt)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA8, x, y, z, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)

    def _refresh_matcap(self, name: str) -> None:
        if name == self._matcap_name or not self._matcap_tex:
            return
        mc = render3d.make_matcap(name, 256)
        h, w = mc.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(mc))
        self._matcap_name = name

    def _init_cube(self) -> None:
        self._cube_prog = self._build_program(_CUBE_VERT, _CUBE_FRAG)
        geo = render3d.cube_geometry()
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
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(24))
        GL.glBindVertexArray(0)
        atlas = make_cube_atlas(96)
        self._cube_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cube_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, atlas.shape[1], atlas.shape[0],
                        0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(atlas))

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

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintGL(self) -> None:  # noqa: N802
        if not self._gl_ok:
            return
        scene = self.ctx.scene
        bg = QColor(self.ctx.theme.background)
        bg3 = (bg.redF(), bg.greenF(), bg.blueF())
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        dpr = self.devicePixelRatioF()
        GL.glViewport(0, 0, int(w * dpr), int(h * dpr))
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClearColor(bg3[0], bg3[1], bg3[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self._tex:
            return
        rs = scene.render
        values = render3d.values_for(rs)
        u_vals = render3d.uniform_values(values)
        self._refresh_matcap(render3d.LIGHTINGS[int(values.get("light", 0)) % len(render3d.LIGHTINGS)])
        inv_vp, rot, cube_rot, flip = self._matrices()

        planes = self._planes()
        n_max = render3d.MAX_CLIP_PLANES
        clip_normals = np.zeros((n_max, 3), np.float32)
        clip_depths = np.zeros(n_max, np.float32)
        clip_thicks = np.zeros(n_max, np.float32)
        for i, (normal, depth, thick) in enumerate(planes):
            clip_normals[i] = normal
            clip_depths[i] = depth
            clip_thicks[i] = thick
        light = render3d.light_dir_view(values.get("lightaz", 0.0), values.get("lightel", 0.0))
        cursor_tex, cross_width, cross_rgb = self._crosshair_uniforms()

        GL.glUseProgram(self._prog)
        u = lambda n: GL.glGetUniformLocation(self._prog, n)  # noqa: E731
        GL.glUniformMatrix4fv(u("uInvViewProj"), 1, GL.GL_TRUE, inv_vp)
        GL.glUniformMatrix3fv(u("uNormalMatrix"), 1, GL.GL_TRUE, rot)
        GL.glUniform3f(u("uBoxHalf"), *map(float, self._box_half))
        GL.glUniform3f(u("uTexSize"), *map(float, self._tex_dims))
        GL.glUniform3f(u("uBg"), *bg3)
        GL.glUniform3f(u("uLightDir"), *light)
        GL.glUniform1i(u("uEffect"), int(render3d.EFFECT_FX.get(rs.effect, 1)))
        GL.glUniform1i(u("uIsRGB"), 1 if self._is_rgb else 0)
        GL.glUniform1f(u("uThreshLo"), u_vals["lo"])
        GL.glUniform1f(u("uThreshHi"), u_vals["hi"])
        GL.glUniform1f(u("uDensity"), u_vals["density"])
        GL.glUniform1f(u("uBrighten"), u_vals["brighten"])
        GL.glUniform1f(u("uSurface"), u_vals["surface"])
        GL.glUniform1f(u("uAmbient"), u_vals["ambient"])
        GL.glUniform1f(u("uDiffuse"), u_vals["diffuse"])
        GL.glUniform1f(u("uSpecular"), u_vals["specular"])
        GL.glUniform1f(u("uShininess"), u_vals["shininess"])
        GL.glUniform1f(u("uBoundThresh"), u_vals["boundthresh"])
        GL.glUniform1f(u("uEdgeThresh"), u_vals["edgethresh"])
        GL.glUniform1f(u("uEdgeMix"), u_vals["edgemix"])
        GL.glUniform1f(u("uColorTemp"), u_vals["colortemp"])
        GL.glUniform1f(u("uGradientMix"), u_vals["gradientmix"])
        GL.glUniform1f(u("uIntensityMix"), u_vals["intensitymix"])
        GL.glUniform1f(u("uHardness"), u_vals["hardness"])
        GL.glUniform1i(u("uPeel"), int(values["peel"]))
        GL.glUniform1f(u("uTlow"), u_vals["tlow"])
        GL.glUniform1f(u("uThigh"), u_vals["thigh"])
        GL.glUniform1i(u("uSteps"), int(values["quality"]))
        GL.glUniform1i(u("uClipCount"), len(planes))
        GL.glUniform1i(u("uClipCutaway"), 1 if rs.cut_away else 0)
        GL.glUniform3fv(u("uClipNormal"), n_max, np.ascontiguousarray(clip_normals))
        GL.glUniform1fv(u("uClipDepth"), n_max, np.ascontiguousarray(clip_depths))
        GL.glUniform1fv(u("uClipThick"), n_max, np.ascontiguousarray(clip_thicks))
        GL.glUniform1i(u("uSliceOverlay"), 1 if values.get("overlay", 0) >= 0.5 else 0)
        GL.glUniform1f(u("uSliceDepth"), u_vals["overlaydepth"])
        GL.glUniform1i(u("uHasOverlay"), 1 if self._ov_tex else 0)
        GL.glUniform1f(u("uSeeThrough"), float(u_vals.get("seethrough", 0.4)))
        GL.glUniform1i(u("uCrosshair"), 1 if cursor_tex is not None else 0)
        GL.glUniform3f(u("uCursor"), *(cursor_tex if cursor_tex is not None else (0.0, 0.0, 0.0)))
        GL.glUniform3f(u("uCrossColor"), *cross_rgb)
        GL.glUniform1f(u("uCrossWidth"), cross_width)
        mats, lens, colours, n_shapes = self._shape_uniforms(values)
        GL.glUniform1i(u("uShapeCount"), n_shapes)
        GL.glUniformMatrix4fv(u("uShapeUnit"), render3d.MAX_SHAPES, GL.GL_TRUE,
                              np.ascontiguousarray(mats, np.float32))
        GL.glUniform3fv(u("uShapeLen"), render3d.MAX_SHAPES, np.ascontiguousarray(lens, np.float32))
        GL.glUniform4fv(u("uShapeColor"), render3d.MAX_SHAPES,
                        np.ascontiguousarray(colours, np.float32))
        GL.glUniform1f(u("uShapeWidth"), 2.5 * self._box_units_per_px())
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._tex)
        GL.glUniform1i(u("uVol"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glUniform1i(u("uMatcap"), 1)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._ov_tex or self._ov_empty)
        GL.glUniform1i(u("uOverlay"), 2)
        use_lut = self._bind_lut(3)
        if not use_lut:
            # A sampler2D on unit 3 must still have something bound.
            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._matcap_tex)
        GL.glUniform1i(u("uLut"), 3)
        GL.glUniform1i(u("uUseLut"), 1 if use_lut else 0)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

        if scene.display.labels and self._cube_prog:
            self._draw_cube(cube_rot, dpr, flip)

    def _matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
        """The camera as matrices: inverse view-projection (the ray caster's
        ``uInvViewProj``), the normal matrix, the cube's rotation, the flip.
        Shared by the paint and the pick, so a click lands where the eye
        looked."""
        cam = self.ctx.scene.render.camera
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        eye = render3d.camera_eye(cam.az, cam.el, cam.dist, cam.target)
        target = np.asarray(cam.target, np.float32)
        view = render3d.look_at(eye, target, np.array([0, 0, 1], np.float32))
        proj = render3d.perspective(self._fovy(), w / h, 0.01, 20.0)
        flip = self.display_flip()
        r4 = np.eye(4, dtype=np.float32)
        r4[0, 0], r4[1, 1], r4[2, 2] = flip
        vm = view @ r4
        inv_vp = np.linalg.inv(proj @ vm).astype(np.float32)
        rot = np.ascontiguousarray(vm[:3, :3], np.float32)
        cube_rot = np.ascontiguousarray(view[:3, :3], np.float32)
        return inv_vp, rot, cube_rot, flip

    def cursor_texcoord(self) -> Optional[np.ndarray]:
        """The crosshair as a coordinate of the render texture (0..1 per
        axis), or None when there is no volume or it lies outside it."""
        if self._geom is None:
            return None
        world = views.cursor_world(self.ctx.store)
        if world is None:
            return None
        affine, dims, _spacing = self._geom
        tex = render3d.texcoord_of_world(np.linalg.inv(affine), dims, world)
        if np.any(tex < -0.01) or np.any(tex > 1.01):
            return None
        return tex

    def _planes(self):
        """The active clip planes as the shader takes them; through the
        crosshair when the cut follows it."""
        rs = self.ctx.scene.render
        at = self.cursor_texcoord() if rs.cut_at_cursor else None
        return render3d.clip_plane_uniforms(self.ctx.scene.clips, at)

    def _fovy(self) -> float:
        """Vertical field of view: 45 degrees, widened in a tile taller than
        it is wide so the HORIZONTAL view stays 45 degrees and the volume
        stays in frame (a narrow tile of a row layout cut the head off)."""
        aspect = max(self.width(), 1) / max(self.height(), 1)
        if aspect >= 1.0:
            return 45.0
        return float(np.degrees(2.0 * np.arctan(np.tan(np.radians(22.5)) / aspect)))

    def _box_units_per_px(self) -> float:
        """How long one screen pixel is in box units, at the camera's
        distance: what keeps a line the same width at any zoom."""
        dist = float(self.ctx.scene.render.camera.dist)
        return 2.0 * dist * np.tan(np.radians(self._fovy()) / 2.0) / max(self.height(), 1)

    def _shape_uniforms(self, values):
        """The shape layers (MRS voxels) drawn in 3-D: their matrices into
        their unit cubes, edge lengths, colours and count."""
        n = render3d.MAX_SHAPES
        colours = np.zeros((n, 4), dtype=np.float32)
        if self._geom is None or values.get("layers", 1.0) < 0.5:
            return np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)), np.zeros((n, 3)), colours, 0
        found = views.shape_layers(self.ctx.store, in_3d=True)[:n]
        affine, dims, _sp = self._geom
        mats, lens = render3d.shape_uniforms([src.box for _l, src in found], affine, dims,
                                             self._box_half)
        for i, (layer, _src) in enumerate(found):
            r, g, b, a = views.shape_colour(layer)
            colours[i] = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        return mats, lens, colours, len(found)

    def _crosshair_uniforms(self):
        """The cursor as a texture coordinate, the line's width in box units
        and its colour; ``(None, 0, colour)`` when the crosshair is off or
        outside the volume. The width is set in screen pixels (the setting's
        thickness, as the slices draw it) at the camera's distance, so a
        line reads the same at any zoom."""
        cs = self.ctx.settings.crosshair
        color = QColor(cs.color)
        if not color.isValid():
            color = QColor(self.ctx.theme.crosshair)
        rgb = (color.redF(), color.greenF(), color.blueF())
        tex = self.cursor_texcoord() if self.ctx.scene.display.crosshair else None
        if tex is None:
            return None, 0.0, rgb
        per_px = self._box_units_per_px()
        width = (max(1, int(cs.thickness)) + 1.0) * 1.5 * per_px
        return (float(tex[0]), float(tex[1]), float(tex[2])), float(width), rgb

    def pick(self, x: float, y: float) -> bool:
        """Move the crosshair to the surface under widget pixel (x, y).

        The march the shader does, done again on the CPU against the uploaded
        8-bit volume, with the same thresholds and clip planes, so what the
        eye sees is what is picked. False when nothing is under the pointer.
        """
        if self._pending is None or self._geom is None:
            return False
        u8 = self._pending[0]
        w, h = max(self.width(), 1), max(self.height(), 1)
        nx, ny = 2.0 * x / w - 1.0, 1.0 - 2.0 * y / h
        ro, rd = render3d.ray_from_ndc(self._matrices()[0], nx, ny)
        rs = self.ctx.scene.render
        values = render3d.values_for(rs)
        u_vals = render3d.uniform_values(values)
        fx = render3d.EFFECT_FX.get(rs.effect, 1)
        tex = render3d.pick_depth(
            u8, self._box_half, ro, rd, lo=u_vals["lo"], hi=u_vals["hi"],
            density=u_vals["density"], clips=self._planes(), cut_away=rs.cut_away,
            steps=min(int(values.get("quality", 512)), 512), surface=fx not in (3, 4),
        )
        if tex is None:
            return False
        affine, dims, _sp = self._geom
        world = render3d.world_of_texcoord(affine, dims, tex)
        self.ctx.run("cursor.set_world", x=float(world[0]), y=float(world[1]), z=float(world[2]))
        return True

    def _draw_cube(self, rot: np.ndarray, dpr: float, flip) -> None:
        flip = tuple(float(v) for v in flip)
        if flip != self._cube_flip and self._cube_tex:
            atlas = make_cube_atlas(96, flip)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._cube_tex)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, atlas.shape[1], atlas.shape[0],
                            0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(atlas))
            self._cube_flip = flip
        s = int(96 * dpr)
        GL.glViewport(int(8 * dpr), int(8 * dpr), s, s)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self._cube_prog)
        proj = render3d.ortho(1.0, 1.0, -4.0, 4.0)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._cube_prog, "uProj"), 1,
                              GL.GL_TRUE, np.ascontiguousarray(proj, np.float32))
        GL.glUniformMatrix3fv(GL.glGetUniformLocation(self._cube_prog, "uRot"), 1,
                              GL.GL_TRUE, np.ascontiguousarray(rot, np.float32))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cube_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._cube_prog, "uAtlas"), 0)
        GL.glBindVertexArray(self._cube_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._cube_nverts)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _mods(self, event) -> set[str]:
        m = event.modifiers()
        mods = set()
        if m & Qt.KeyboardModifier.ShiftModifier:
            mods.add("shift")
        if m & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            mods.add("ctrl")
        if m & Qt.KeyboardModifier.AltModifier:
            mods.add("alt")
        return mods

    def _clip_index(self) -> int:
        """The clip plane gestures act on: the one the panel has selected."""
        return int(getattr(self.ctx, "active_clip", 0))

    def _clip_active(self) -> bool:
        clips = self.ctx.scene.clips
        i = self._clip_index()
        return bool(clips and i < len(clips) and clips[i].active)

    def _focus_viewer(self) -> None:
        w = self.parentWidget()
        while w is not None and not getattr(w, "_is_viz_viewer", False):
            w = w.parentWidget()
        if w is not None:
            w.setFocus(Qt.FocusReason.MouseFocusReason)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._focus_viewer()
        button = {Qt.MouseButton.LeftButton: "left", Qt.MouseButton.RightButton: "right",
                  Qt.MouseButton.MiddleButton: "middle"}.get(event.button())
        if button is None:
            return
        tool = inputmap.lookup("render", button, self._mods(event), self.ctx.settings.mousemap)
        if tool == "clip_tilt" and not self._clip_active():
            tool = inputmap.lookup("render", button, set(), self.ctx.settings.mousemap)
        self._tool = tool
        self._last = event.position().toPoint()
        self._press = self._last
        self._moved = False
        if tool == "pick":
            self.pick(event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._last is None:
            return
        pos = event.position().toPoint()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if self._press is not None and (abs(pos.x() - self._press.x()) > 3
                                        or abs(pos.y() - self._press.y()) > 3):
            self._moved = True
        if self._tool == "orbit":
            self.ctx.run("render.orbit", d_az=dx * 0.01, d_el=dy * 0.01)
        elif self._tool == "pan":
            self.ctx.run("render.pan", dx=float(dx), dy=float(dy))
        elif self._tool == "zoom":
            self.ctx.run("render.zoom", steps=-dy / 40.0)
        elif self._tool == "clip_tilt":
            self.ctx.run("clip.tilt", d_az=dx * 0.6, d_el=-dy * 0.6, index=self._clip_index())
        elif self._tool == "pick":
            self.pick(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        # A click that never dragged is a pick: the crosshair goes to the
        # surface under the pointer, as a click on a slice moves it there.
        if (self._tool == "orbit" and not self._moved and self._press is not None
                and self.ctx.scene.display.crosshair):
            self.pick(float(self._press.x()), float(self._press.y()))
        self._last = None
        self._press = None
        self._tool = "none"

    def wheelEvent(self, event) -> None:  # noqa: N802
        ad, pd = event.angleDelta(), event.pixelDelta()
        a = ad.y() or ad.x()
        steps = (a / 120.0) if a != 0 else ((pd.y() or pd.x()) / 320.0)
        if steps == 0.0:
            return
        horizontal = (ad.x() != 0 and ad.y() == 0) or (pd.x() != 0 and pd.y() == 0)
        gesture = "hwheel" if horizontal else "wheel"
        tool = inputmap.lookup("render", gesture, self._mods(event), self.ctx.settings.mousemap)
        if tool == "clip_push" and not self._clip_active():
            tool = "zoom"
        if tool == "clip_push":
            self.ctx.run("clip.nudge", delta=0.02 * steps, index=self._clip_index())
        elif tool == "zoom":
            self.ctx.run("render.zoom", steps=steps)
        event.accept()

    def grab_image(self) -> QImage:
        return self.grabFramebuffer()


__all__ = ["RenderCanvas", "canonical_geometry", "gpu_available", "make_cube_atlas",
           "prepare_overlays", "prepare_upload", "request_gl_format"]
