import numpy as np
import pytest

from autofocus.test_utils import VLaser


xs = np.linspace(-5, 5, 10000)
dx = xs[1] - xs[0]

xygrid = np.mgrid[-5:5:0.01, -5:5:0.01]


def test_vlaser_normalised():
    narrow = VLaser(x0=0, y0=0, z0=0, width=0.1)
    wide = VLaser(x0=0, y0=0, z0=0, width=1)
    vwide = VLaser(x0=0, y0=0, z0=0, width=2)

    assert np.trapz(narrow(xs, ys=0, sz=0), xs) == \
           pytest.approx(np.trapz(wide(xs, ys=0, sz=0), xs)) == \
           pytest.approx(np.trapz(vwide(xs, ys=0, sz=0), xs))
    

def test_vlaser_width():
    l = VLaser(x0=0, y0=0, z0=0, width=0.5)
    l_width = VLaser.find_FWHM(xs, l(xs, ys=0, sz=0)) 

    assert l_width == pytest.approx(0.5870 / 2, abs=2 * dx)

    dl = VLaser(x0=0, y0=0, z0=0, width=1)
    dl_width = VLaser.find_FWHM(xs, dl(xs, ys=0, sz=0))

    # scales appropriately
    assert dl_width == pytest.approx(2 * l_width, abs=2 * dx)


def test_vlaser_defocus():
    l = VLaser(x0=0, y0=0, z0=0, width=1)

    focus = l(xs, 0, sz=0)
    underfocus = l(xs, 0, sz=-1)
    overfocus = l(xs, 0, sz=1)

    # check spot is wider when out of focus
    assert VLaser.find_FWHM(xs, underfocus) > VLaser.find_FWHM(xs, focus)
    assert VLaser.find_FWHM(xs, overfocus) > VLaser.find_FWHM(xs, focus)

    # equal defocusses should be equal
    assert underfocus == pytest.approx(overfocus)


def test_vlaser_offcentre():
    l = VLaser(x0=1, y0=0, z0=0, width=1)

    # 1D
    # x coordinate of max of y!
    assert xs[np.argmax(l(xs, ys=0, sz=0))] == pytest.approx(1, abs=2 * dx)

    # 2D
    l = VLaser(x0=1, y0=1.5, z0=0, width=1)

    gxs, gys = xygrid
    data = l(gxs, gys, sz=0)

    # 2D equivalent of getting the coords!
    ixmax, iymax = np.unravel_index(np.argmax(data, axis=None), data.shape)

    assert gxs[ixmax, 0] == pytest.approx(1)
    assert gys[0, iymax] == pytest.approx(1.5)
    