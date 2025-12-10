# AutoFocus

This repo contains the code for adding a gradient ascent-based autofocussing proceedure to a setup.

This code originates from the [UDNS Lab](https://www.ultrafastdynamics.com/) at TU Eindhoven.

## Installation

Clone this respository:

```
git clone https://github.com/0Hughman0/autofocus.git
```

Then in your environment, e.g. conda environment, navigate to the directory with `pyproject.toml` in and run:

```
pip install .
```

This will install the `autofocus` package, and its dependencies.

After installation, the `autofocus` package will be available for you to import and utilise in your code.

## Usage

To adapt the code to run on your setup, you will need to implement 'driver' classes that `autofocus` can use.

Provided below is an example used in our setups:

```
class PumpCtrl(PumpCtrlABC):
    """
    Wraps the Control_xyz_stage driver to allow it to be used by AutoFocus.
    """

    def __init__(self, stage: 'Control_xyz_stage'):
        super().__init__()

        self.stage = stage

    def get_z(self):
        x, y, z = self.stage.get_position()
        return z
    
    def go(self, z):
        self.stage.set_position(z=z)
        return self.get_z()
    
    def shift(self, dz):
        x, y, z = self.stage.get_position()
        z += dz
        self.stage.set_position(z=z)
        return self.get_z()


class ProbeCtrl(ProbeCtrlABC):
    """
    Wraps Control_ScanningMirror to allow it to be used by AutoFocus.
    """

    def __init__(self, mirror: 'Control_ScanningMirror'):
        super().__init__()

        self.mirror = mirror
    
    def get_xy(self):
        x, y = self.mirror.get_position(output=True)
        return x, y
    
    def go(self, x, y):
        self.mirror.set_position(x=x if x else None, y=y if y else None)
        return self.get_xy()
    
    def shift(self, dx, dy):
        x, y = self.get_xy()

        x += dx
        y += dy

        return self.go(x=x if dx else None, y=y if dy else None)


def wrap_lockin(lockin: 'Control_LI', wait=0, negate=True, **poll_kwargs) -> ResponseFunc:
    """
    Wrap the Control_LI to allow it to be used by AutoFocus.
    """
    def get_response(x, y, z):
        time.sleep(wait)
        if negate:
            return -lockin.read_signal(**poll_kwargs)[0]  # sometimes it's negative!
        else: 
            return lockin.read_signal(**poll_kwargs)[0]

    return get_response
```

With these drivers, the autofocus can be used as follows:

```
from ustm.drivers import (
        Control_LI, Control_ScanningMirror, Control_xyz_stage, ProbeCtrl, PumpCtrl, wrap_lockin
    )

from autofocus import do_autofocus

with (
        Control_LI() as li, Control_ScanningMirror(li) as mirror, 
        Control_xyz_stage('Axis 1', 'Axis 2', 'Axis 3') as stage
    ):

    for stage, p, dxdydz, I in do_autofocus(
        ProbeCtrl(mirror), # Allows autofocus to control the mirror.
        PumpCtrl(stage), # Allows autofocus to control the xyz stage.
        wrap_lockin(li, wait=0.1, poll_time=1, negate=True), # uses these parameters when polling the lockin.
        dr=0.05 # other parameters e.g. the step size used when calculating gradients.
    ):
        # stage is the current stage
        # p is the position it moved to
        # dxdydz is the last step
        # I is the last intensity.
        print(stage, p, dxdydz, I)

        # If you want to interupt the autofocus, just break out of this loop! e.g.
        if I > 1000000000:
            # laser exploding
            break
```

