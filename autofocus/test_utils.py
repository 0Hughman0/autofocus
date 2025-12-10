from typing import Callable, Any, TypeVar, Tuple
from typing_extensions import Protocol, Iterable

import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as tnp

from .autofocus import ProbeCtrlABC, PumpCtrlABC, AutoFocus


class VLaser:
    NA = 0.67
    angle = np.arcsin(NA / 1)  # I think this is the right way to do this!

    def __init__(self, x0=0, y0=0, z0=0, width=1, A=1e3):
        """
        Focal point is x0, y0, z0 with width and A max intensity

        ... it's normalised!
        """
        
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.width = width
        self.A = A

    def __call__(self, xs, ys, sx=0, sy=0, sz=0):
        """
        xs, ys are the grid the laser profile is built on.

        sx, sy, sz are the position of the stage it's projected on.
        """
        defocus = sz - self.z0
        xs = xs - self.x0
        ys = ys - self.y0
        actual_width = self.width + abs(np.tan(self.angle) * defocus)
        return (self.A / (actual_width * 3927)) * np.exp(-8 * (xs **2 + ys ** 2) / (actual_width * actual_width))

    @staticmethod
    def find_FWHM(xs, ys):
        HM = 0.5 * np.max(ys)

        above = xs[ys > HM]

        lower = np.min(above)
        upper = np.max(above)

        return upper - lower
    
    def marker(self, ax, xs, ys, sx=0, sy=0, sz=0):    
        ax.plot(self.y0, self.x0, 'x')


Grid = TypeVar('Grid')


class VCamera:

    def __init__(self, 
                 get_stage_xyz: Callable[[], Tuple[float, float, float]], 
                 imagers=Iterable[Callable[[Grid, Grid, float, float, float], Grid | None]],
                 drawers=Iterable[Callable[[Axes, Grid, Grid, float, float, float], Grid | None]], 
                 x0=-5, x1=5, y0=-5, y1=5, dx=0.01, dy=0.01):
        
        self.get_stage_xyz = get_stage_xyz
        self.imagers = imagers
        self.drawers = drawers
        
        self.xygrid = np.mgrid[x0:x1:dx, y0:y1:dy]

        self.fig = None
        self.im = None

    def plot(self):
        self.fig = plt.figure()
        ax = self.ax = self.fig.add_subplot(111)
        

        xs, ys = self.xygrid
        sx, sy, sz = self.get_stage_xyz()
        extent = mgrid2extent(xs, ys)

        Is = np.zeros_like(ys)
        
        for imager in self.imagers:
            I = imager(xs, ys, sx, sy, sz)
            
            if I is not None:
                Is += I

        self.im = ax.imshow(Is, extent=extent, origin='lower')

        for drawer in self.drawers:
            drawer(ax, xs, ys, sx, sy, sz)

        plt.show()

    def update(self):
        xs, ys = self.xygrid
        sx, sy, sz = self.get_stage_xyz()

        Is = np.zeros_like(ys)

        for line in self.fig.axes[0].lines:
            line.remove()
        
        for imager in self.imagers:
            I = imager(xs, ys, sx, sy, sz)
            
            if I is not None:
                Is += I

        self.im.set_data(Is)
        self.im.autoscale()

        for drawer in self.drawers:
            drawer(self.ax, xs, ys, sx, sy, sz)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class VFocus(PumpCtrlABC):

    def __init__(self, iz=0., cbs=None):
        self._z = iz

        if cbs is None:
            cbs = []
        self.cbs = cbs

    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, val):
        self._z = val

        for cb in self.cbs:
            cb(self._z)

    def get_z(self):
        return self.z

    def shift(self, dz: float) -> float:
        self.z += dz
        return self.z
    
    def go(self, z):
        self.z = z

        return z
    


class VProbeMirror(ProbeCtrlABC):

    def __init__(self, probe: VLaser, ix=0., iy=0., cbs=None):
        self.probe = probe
        self._x = ix
        self._y = iy

        if cbs is None:
            cbs = []

        def move_probe(x, y):
            self.probe.x0 = x
            self.probe.y0 = y

        cbs.append(move_probe)

        self.cbs = cbs
        move_probe(self.x, self.y)

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, val):
        self._x = val

        for cb in self.cbs:
            cb(self.x, self.y)
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, val):
        self._y = val

        for cb in self.cbs:
            cb(self.x, self.y)

    def get_xy(self):
        return self.x, self.y

    def shift(self, dx, dy):
        self.x += dx
        self.y += dy

        return self.x, self.y
    
    def go(self, x, y):
        self.x = x
        self.y = y

        return x, y


class CostFunc(Protocol):
    def __call__(self, pump_I: np.ndarray, probe_I: np.ndarray) -> float:
        """
        Return signal proptional to the expected response for a given pump and probe
        intensity distribution.
        """
        pass


def focus_sensitive(pump_I, probe_I) -> float:
    """
    If we only make it the sum of product, we don't get an inscentive to 
    maximize focus... which is interesting, but no doubt in the maths!
    """
    return np.trapz(np.trapz(pump_I ** 1.1 *  probe_I ** 1.1, axis=0, dx=0.01), dx=0.01)


class VLockin:
    xygrid = np.mgrid[-3:3:0.01, -3:3:0.01]

    pump: VLaser
    probe: VLaser

    def __init__(self, pump: VLaser, probe: VLaser, focus: VFocus, probe_mirror: VProbeMirror,
                 cost_func: CostFunc | None = None, noise=0):
        self.pump = pump
        self.probe = probe

        self.focus = focus
        self.probe_mirror = probe_mirror

        if cost_func is None:
            self.cost_func: CostFunc = focus_sensitive
        else:
            self.cost_func = cost_func

        self.noise = noise

    def poll(self) -> float:
        focus_z = self.focus.get_z()
        xs, ys = self.xygrid

        noise = self.noise * np.random.random(xs.shape)

        pump = self.pump(xs, ys, sz=focus_z) * (1 + noise)
        probe = self.probe(xs, ys, sz=focus_z) * (1 + noise)
        
        return self.cost_func(pump, probe)


class TAutoFocus(AutoFocus):
    
    def __init__(self, probe_ctrl, pump_ctrl, get_response,
                       pump: VLaser=None,
                       probe: VLaser=None,
                       lockin: VLaser=None):
        super().__init__(probe_ctrl, pump_ctrl, get_response)

        self.pump = pump
        self.probe = probe
        self.lockin = lockin

    def plot_camera(self, which='both', markers=True):
        match (which):
            case 'both':
                imagers = [self.pump, self.probe]
            case 'pump':
                imagers = [self.pump]
            case 'probe':
                imagers = [self.probe]

        if markers:
            drawers = [laser.marker for laser in imagers]
        else:
            drawers = []
            
        camera = VCamera(
            get_stage_xyz=self.positions,
            imagers=imagers,
            drawers=drawers
        )

        camera.plot()

        return camera
