"""The 3-D viewer actually drawing something, on a real OpenGL context.

Everything else about the 3-D viewer is tested through ``test_nifti_3d_view``,
which runs under Qt's ``offscreen`` platform and forces the GPU gate open
because that platform cannot create an OpenGL context at all. Those tests cover
the wiring: which button enables which control, what reaches the GL widget.

They cannot cover whether a single pixel is drawn, and that is the half most
likely to break, because a shader that fails to compile fails silently and the
wiring is unaffected.

So these ask for a REAL context, off screen but not offscreen: a QOffscreenSurface
with a QOpenGLContext, which the platform plugin has to supply. On a machine
without one, or under the offscreen platform, they skip and say why. Run them on
a desktop:

    pytest tests/gui/test_nifti_3d_render.py            # real platform, real GL

They are the reason the suite can be run headless without pretending the GPU
path is covered.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("OpenGL")

pytestmark = pytest.mark.gui


@pytest.fixture
def gl_context(qapp):
    """A real OpenGL 3.3 core context, or a skip explaining why not."""
    from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext
    from bidsmgr.gui.widgets.nifti_gl_view import request_gl_format

    surface = QOffscreenSurface()
    surface.setFormat(request_gl_format())
    surface.create()
    if not surface.isValid():
        pytest.skip("the Qt platform plugin cannot create an offscreen surface")

    context = QOpenGLContext()
    context.setFormat(request_gl_format())
    if not context.create() or not context.makeCurrent(surface):
        pytest.skip("no OpenGL 3.3 core context on this machine")

    yield context
    context.doneCurrent()


def test_the_gate_agrees_with_reality(gl_context):
    """If a context exists, the gate must open, or the 3-D view is hidden from
    a user who has a perfectly good GPU."""
    from bidsmgr.gui.widgets.nifti_gl_view import gpu_available

    assert gpu_available()


def test_a_volume_renders_something(qtbot, gl_context):
    """A shader that fails to compile fails silently: the wiring is intact, the
    controls all work, and the image is empty. Only pixels prove otherwise."""
    from PyQt6.QtWidgets import QApplication
    from bidsmgr.gui.widgets.nifti_gl_view import RaycastGLWidget

    # A bright cube in the middle of an otherwise empty volume.
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume[8:24, 8:24, 8:24] = 1.0

    widget = RaycastGLWidget()
    qtbot.addWidget(widget)
    widget.resize(128, 128)
    widget.set_volume(volume, (1.0, 1.0, 1.0))
    widget.show()
    qtbot.waitExposed(widget)
    QApplication.processEvents()

    image = widget.grabFramebuffer()
    assert not image.isNull(), "the widget produced no framebuffer at all"

    # Something must be brighter than the background, or nothing was drawn.
    brightest = max(
        image.pixelColor(x, y).lightness()
        for x in range(0, image.width(), 4)
        for y in range(0, image.height(), 4)
    )
    assert brightest > 10, "the volume rendered to an empty image"
