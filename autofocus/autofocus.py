from typing import Tuple, Callable, Sequence, Literal, Generator, TypeVar
from abc import ABC
import logging

import numpy as np

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


T = TypeVar('T')

def damp(delta: T, asym: float) -> T:
    coef = asym * 2 / np.pi
    return coef * np.arctan(delta / coef)


class PumpCtrlABC(ABC):
    """
    Describes the form the PumpCtrl parameter that's passed to autofocus should have.

    See concrete implementations in drivers.

    This means, if we start using a different scanning mirror or xyz stage, we just have to re-implement 
    PumpCtrl and LockinCtrl for AutoFocus to still work.
    """

    def get_z(self) -> float:
        """
        Get the z position of the stage (which controls the focus of the Pump).
        """
        pass

    def shift(self, dz: float) -> float:
        """
        Ask the stage to shift the focus, return the new position.
        """
        pass

    def go(self, z: float) -> float: 
        """
        Go to a z position
        """
        pass


class ProbeCtrlABC(ABC):

    def get_xy(self) -> Tuple[float, float]:
        """
        Get the x, y position of the probe, i.e. from the scanning mirror.
        """
        pass

    def shift(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        Shift the probe by dx and dy. Return the new position.
        """
        pass

    def go(self, x: float, y: float) -> Tuple[float, float]:
        pass


ResponseFunc = Callable[[float, float, float], float]


def first_initial_done(auto: 'AutoFocus', p, dxdydz, I):
    """
    Returns Ture (i.e. is done) when 5 steps have been performed
    """
    return np.sum(np.array(auto.past_stages) == auto.stage) > 5


def coarse_xy_done(auto, p, dxdydz, I):
    return I > (auto.min_thresh / 2)


def fine_xy_done(auto, p, dxdydz, I):
    dx, dy, dz = dxdydz
    return I > auto.min_thresh and abs(dx) < auto.min_step and abs(dy) < auto.min_step


def central_xy_done(auto: 'AutoFocus', p, dxdydz, I):
    lookback = auto.central_lookback
    
    if len(auto.recent_shifts) < lookback:
        return False

    recent_shifts = np.array(auto.recent_shifts)
    recent_signs = np.sign(recent_shifts)

    done = np.all(np.abs(np.diff(recent_signs, axis=0))[:, :2] >= 1)
    
    if done:
        xy_centre = np.mean(auto.past_positions[-lookback:], axis=0)
        logger.info(f"Moving to average xy centre {xy_centre}")
        auto.go(xy_centre[0], xy_centre[1], None)

    return done


def coarse_focus_done(auto, p, dxdydz, I):
    return I > (auto.min_thresh)


def fine_focus_done(auto, p, dxdydz, I):
    dx, dy, dz = dxdydz
    return I > auto.min_thresh and abs(dz) < auto.min_step


def central_focus_done(auto: 'AutoFocus', p, dxdydz, I):
    lookback = auto.central_lookback
    
    if len(auto.recent_shifts) < lookback:
        return False

    recent_shifts = np.array(auto.recent_shifts)
    recent_signs = np.sign(recent_shifts)

    # checks if the sign flipped successively in recent_signs
    done = np.all(np.abs(np.diff(recent_signs, axis=0))[:, 2] >= 1)
    
    if done:
        xy_centre = np.mean(auto.past_positions[-lookback:], axis=0)
        logger.info(f"Moving to average xy centre {xy_centre}")
        auto.go(None, None, xy_centre[2])
    
    return done


def central_focus_I_done(auto: 'AutoFocus', p, dxdydz, I):
    lookback = auto.central_lookback
    
    if len(auto.recent_Is) < lookback:
        return False

    recent_Is = np.array(auto.recent_Is)
    
    # checks all Is are within threshhold of mean.
    combinations = np.meshgrid(recent_Is, recent_Is)
    diffs = np.diff(combinations, axis=0)
    average = np.mean(recent_Is)

    done = np.all((diffs / average) <= auto.flat_thresh)
    
    if done:
        xy_centre = np.mean(auto.past_positions[-lookback:], axis=0)
        logger.info(f"Moving to average xy centre {xy_centre}")
        auto.go(None, None, xy_centre[2])
    
    return done


def central_xy_I_done(auto: 'AutoFocus', p, dxdydz, I):
    lookback = auto.central_lookback
    
    if len(auto.recent_Is) < lookback:
        return False

    recent_Is = np.array(auto.recent_Is)
    
    # checks all Is are within threshhold of mean.
    combinations = np.meshgrid(recent_Is, recent_Is)
    diffs = np.diff(combinations, axis=0)
    average = np.mean(recent_Is)

    done = np.all((diffs / average) <= auto.flat_thresh)

    if done:
        xy_centre = np.mean(auto.past_positions[-lookback:], axis=0)
        logger.info(f"Moving to average xy centre {xy_centre}")
        auto.go(xy_centre[0], xy_centre[1], None)
    
    return done


class AutoFocus:
    """
    A class for performing an autofocus proceedure.

    Broadly, the class steps through different stages. At each step, it moves a small amount, and checks if the signal goes 
    up or down, and from that, moves in the direction that should increase the signal.
    
    It is now recommended to use the `do_autofocus` function, defined below, rather than creating an AutoFocus instance directly.

    Arguments
    ---------
    probe_ctrl: ProbeCtrlABC
        The object autofocus uses to control the probe. It should have the methods described in `ProbeCtrlABC`.
    pump_ctrl: PumpCtrlABC
        The object autofocus uses to control the pump. It should have the methods described in `PumpCtrlABC`.
    get_response: ResponseFunc
        The function called to get the signal/ response at a given time. This is the parameter autofocus is trying to maximize.
    width: float
        This is the approx. width of the beam. *In practice this is used to set the maximum step size*. The algorithm will never
        step more than this value to try to increase the signal. This ensures it doesn't get over-excited and step over the optimal focus.
    rate: float, [float, float, float]
        The step size is calculated by multiplying the gradient by the rate, along with coarsening factors. The rate is stored as a numpy array
        of length 3. Where each float corresponds to [probe x, probe y, focus z]. If a single rate value is provided, this is used for
        each dimension i.e. set to [rate, rate, rate].
    min_thresh: float
        The autofocus has coarse stages for xy and focus which take larger steps vs the gradient dr. The autofocus will not move on from coarse
        stages, unless the signal is over the min_thresh.
    min_step: float
        In the fine xy/focus stages, as you get close to the best alignment (and also due to a decaying feature in `self.rate`) the step sizes
        will get quite small, and so this is a sign you are close to well focussed. min_step is the value at which autofocus considers the 
        step sufficiently small, it thinks it's close to best focus, and so will move on the central xy/ focus stage.
    dr: float
        dr is the amount the scanning mirror and xyz stage move when calculating the gradient. Because the alignment sensitivity is less for 
        the z direction, the dr in z is actually multiplied by `Autofocus.z_dr_multiplier`, which by default is 5...
    flat_thresh: float
        In the central focus steps, the autofocus will look back at central_lookback number of steps, and if the intensities are within the flat_thresh
        amount, the autofocus will consider the intenisty constant, and considers itself well focussed.
    first_stage: str
        This string represents which stage we are in from `Autofocus.stages`. You can explicitly set this value to a stage and the autofocus will proceed 
        from this stage.
    
    Class Attributes
    ----------------
    stages: dict[str, Callable[[auto: AutoFocus, p: [float, float, float], dxdydz: [float, float, float], I: float], bool]]
        Stages is a dictionary (whose order is important!) which defines the stages the autofocus runs through. The key is the name of the stage. The value 
        is a callable i.e. a function that takes the parameters auto, p, dxdydz and I where auto is the auto object that's running, p is the position of the system
        dxdydz is the last step taken to try and improve the focus and I is the intensity. This function should return True, once, based on the parameters 
        passed to it, it thinks this stage is done. Otherwise, it should return False
    z_dr_multiplier: float
        This is the amount _more_ than dr, to get the gradient in the z direction.
    dr_coarsening_factor: float
        When in a coarse step, the autofocus will take many times larger a step dr when calculating gradients.
    central_lookback: int
        For the central focus steps, this is the amount of steps back to look at, to see if the focussing is leveling off. (See the central_xy_I_done).
    flat_thresh: float
        In the central focus steps, the autofocus will look back at central_lookback number of steps, and if the intensities are within the flat_thresh
        amount, the autofocus will consider the intenisty constant, and considers itself well focussed.

    Attributes
    ----------
    rate: np.array([float, float, float])
        The current rate at which to step.
    _flip: bool
        To minimize drift/ hysterisis when the autofocus is calculating the gradients, it will alternate between stepping forwards (i.e. +autofocus.dr) 
        and backwards i.e. -autofocus.dr... this should help with hysterisis... I think.
    stage: str
        This string represents which stage we are in from `Autofocus.stages`. You can explicitly set this value to a stage and the autofocus will proceed 
        from this stage.

    Examples
    --------
    This class can be used directly in the ST_GUI.

    ```python
    from ustm.autofocus import AutoFocus
    
    from my_setup.drivers import (
        ProbeCtrl, PumpCtrl, make_get_response
    ) # These should implement the ProbeCtrlABC, PumpCtrlABC and create a function that implements get_response.

    with (
            Control_Lockin() as li, Control_ScanningMirror(li) as mirror, 
            Control_xyz_stage('Axis 1', 'Axis 2', 'Axis 3') as stage
        ): # Connect to your devices however you need.
        # Create an autofocus object
        auto = AutoFocus(
            ProbeCtrl(mirror), # Allows autofocus to control the mirror.
            PumpCtrl(stage), # Allows autofocus to control the xyz stage.
            make_get_response(li), # Allows autofocus to get the response.
            dr=0.05 # other parameters e.g. the step size used when calculating gradients.
        )

        # To step through the autofocus, you put it in a for loop!
        for stage, p, dxdydz, I in auto:
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
    """

    stages = {
        'coarse-initial': first_initial_done,
        # 'initial': first_initial_done,
        'coarse-xy': coarse_xy_done,
        'fine-xy': fine_xy_done,
        'central-xy': central_xy_I_done,
        'coarse-focus': coarse_focus_done,
        'fine-focus': fine_focus_done,
        'central-focus': central_focus_I_done,
        'done': lambda: True
    }

    z_dr_multiplier = 5
    z_rate_multiplier = 1
    dr_coarsening_factor = 10
    central_lookback = 4

    def __init__(self, probe_ctrl: ProbeCtrlABC, 
                       pump_ctrl: PumpCtrlABC, 
                       get_response: ResponseFunc,
                       width=1.0,
                       rate: float | Sequence[float]=1e2,
                       min_thresh=1e-5,
                       min_step=0.05,
                       dr=0.05,
                       flat_thresh=5e-2,
                       first_stage: None | Literal[
                            'coarse-initial',
                            'coarse-xy',
                            'fine-xy',
                            'central-xy',
                            'coarse-focus',
                            'fine-focus',
                            'central-focus',
                        ] = None,
                       ) -> None:
        
        self.probe_ctrl = probe_ctrl
        self.pump_ctrl = pump_ctrl
        self.get_response = get_response

        self.rate = rate

        self.width = width
        self.min_thresh = min_thresh
        self.min_step = min_step

        self.dr = np.array([dr, dr, dr * self.z_dr_multiplier])
        self.flat_thresh = flat_thresh
        self.stage: str | None = first_stage

        self._flip = False  # reduce historesis by flipping direction

        self._istages = None
        self._past_stages = []
        self._past_positions = []
        self._past_shifts = []
        self._past_Is = []

    @property
    def rate(self):
        return self._rate
    
    @rate.setter
    def rate(self, value):
        """
        This ensures the rate is always in the right form i.e. np.array([rate x, rate y, rate z]).

        You can set rate as a float, and it will create an array filled with that value.
        """

        if isinstance(value, (np.ndarray)):
            self._rate = value
        
        elif isinstance(value, (tuple, list)):
            self._rate = np.array(value)
        else:
            self._rate = np.array([value, value, value])

    def _n_recent(self):
        """
        Gets the n most recent stages, i.e. previous outcomes from autofocus steps.

        n is determined by `autofocus.central_lookback`.

        If no stages have occured yet i.e. the autofocus hasn't taken a step yet, then it will return None instead.
        """
        recent_stages = self.past_stages[-1:-self.central_lookback - 1:-1]

        index = 0

        for stage in recent_stages:
            if stage != self.stage:
                break

            index += 1 
            
        
        return index or None

    @property
    def past_stages(self) -> list[str]:
        """
        Get the past stages the autofocus has ran through
        """
        return self._past_stages

    @property
    def past_positions(self):
        """
        Get the past positions the autofocus has stepped through
        """
        return self._past_positions
    
    @property
    def past_shifts(self):
        """
        Get the past shifts the autofocus has stepped
        """
        return self._past_shifts

    @property
    def past_Is(self):
        """
        Get the past intensities the autofocus has reached
        """
        return self._past_Is
    
    @property
    def recent_positions(self):
        """
        Get the most recent positions. Where recent is defined by `AutoFocus.central_lookback`.
        """
        if self._n_recent() is None:
            return None
        return self.past_positions[-self._n_recent():]
    
    @property
    def recent_shifts(self):
        """
        Get the most recent shifts. Where recent is defined by `AutoFocus.central_lookback`.
        """
        if self._n_recent() is None:
            return None
        return self.past_shifts[-self._n_recent():]
    
    @property
    def recent_Is(self):
        """
        Get the most recent intensities. Where recent is defined by `AutoFocus.central_lookback`.
        """
        if self._n_recent() is None:
            return None
        return self.past_Is[-self._n_recent():]

    def poll(self):
        """
        Get the intensity from the lock-in
        """
        return self.get_response(*self.positions())

    def positions(self):
        """
        Returns the x, y positions of the probe and the z position of the pump.
        """
        z = self.pump_ctrl.get_z()
        x, y = self.probe_ctrl.get_xy()

        return x, y, z
    
    def go(self, x: float | None, y: float | None, z: float | None):
        """
        Go to a position x, y, z.

        If any of x, y or z are None, then no attempt to move in that coordinate is made.
        """
        if x is not None or y is not None:
            x, y = self.probe_ctrl.go(x, y)
        else:
            x, y = self.probe_ctrl.get_xy()
        
        if z is not None:
            z = self.pump_ctrl.go(z)
        else:
            z = self.pump_ctrl.get_z()

        return x, y, z
        
    def shift(self, dx, dy, dz):
        """
        shift the probe to a new position
        """
        if dx or dy:
            x, y = self.probe_ctrl.shift(dx, dy)
        else:
            x, y = self.probe_ctrl.get_xy()
        
        if dz:
            z = self.pump_ctrl.shift(dz)
        else:
            z = self.pump_ctrl.get_z()

        return x, y, z
                
    def _next_dr(self):
        """
        Calculate what the next steps dr will be used to calculate the next gradient.
        """
        dr = self.dr.copy()

        if self.stage and 'coarse' in self.stage:
            # Exponential?... no, because this value never over-writes self.dr
            dr = dr * self.dr_coarsening_factor

        if self._flip:
            dr = -dr

        if self.stage and 'xy' in self.stage:
            dr[2] = 0
        
        if self.stage and 'focus' in self.stage:
            dr[:2] = 0

        self._flip = not self._flip       

        return dr

    def _grad(self) -> Tuple[float, float, float]:
        """
        Compute the gradient
        """
        drs = self._next_dr()

        I0 = self.poll()

        p0 = self.positions()

        deltas = np.zeros((3, 2))
        deltas[:, :] = I0

        for d, dr in enumerate(drs):
            self.go(*p0)

            if not dr:
                continue

            shift = np.array([0.0, 0.0, 0.0])
            shift[d] = dr
            
            self.shift(*shift)

            deltas[d, 1] = self.poll()

        self.go(*p0)

        grad = np.diff(deltas, axis=1)[:, 0] / drs
        grad = np.where(drs != 0, grad, 0)  # otherwise get inifinity where dr = 0

        logger.debug(f"Gradient: {grad}")

        return grad
    
    def next_step(self) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        """
        Take the next step in trying to optimize the focus.

        This will first compute the gradient.

        Then it will turn this gradient into a step size.

        Then it will dampen the steps to ensure it doesn't overshoot.

        Then it will move the system the appropriate amount to improve the focus.

        Returns
        -------
        p, dxdydz, I: [float, float, float], [float, float, float], float
            p is the position after the step. dxdydz is the change it took this step and I is the intensity after this step.

        """
        grad = self._grad()

        rate = self.rate.copy()

        if self.stage and 'coarse' in self.stage:
            rate = rate * 10

        rate[2] = rate[2] * self.z_rate_multiplier

        dxdydz = grad * self.rate
        dxdydz = damp(dxdydz, self.width)
        
        p = self.shift(*dxdydz)

        return p, tuple(dxdydz), self.poll()
    
    def _prepare_next(self, p: tuple[float, float, float], dxdydz: tuple[float, float, float], I: float):
        """
        Based on the previous step, adapt rate and dr.

        And check if we can move onto the next step... and if we can... then move on to the next step.
        """
        # decay in rate if flipping over
        new_signs = np.sign(dxdydz)

        if self._past_shifts:
            last_signs = np.sign(self._past_shifts[-1])
            # sign changes for a sufficiently high signal indicate we may have hopped over best focus.
            # so we might be overshooting.
            overshooting = (last_signs != new_signs) & (np.array(dxdydz) > self.min_step)
        else: 
            overshooting = np.array([False, False, False])

        # If we're overshooting on a given coordinate, then reduce the rate.
        self.rate = np.where(overshooting, self.rate * 0.75, self.rate)
        self.dr = np.where(overshooting, self.dr * 0.95, self.dr)

        if np.any(overshooting):
            logger.info(f"Detecting overshooting, reducing rate and dr to {self.rate} {self.dr}")
        
        self._past_positions.append(p)
        self._past_shifts.append(dxdydz)
        self._past_Is.append(I)
        self._past_stages.append(self.stage)

        check_done = self.stages.get(self.stage)

        if check_done and check_done(self, p, dxdydz, I):
            self.stage = next(self._istages)

    def __iter__(self):
        """
        This is actually what runs the autofocus...

        This code is ran when you write:

        ```python
        for stage, p, dxdydz, I in auto:
            ...
        ```

        it erases past stages, positions, shifts and Is.

        If you specified a stage, it will start from there.

        Yields
        ------
        stage: str
            The current stage
        p: tuple[float, float, float]
            The position after the step
        dxdydz: tuple[float, float, float]
            The last movement to improve the focus
        I: float
            The intensity after the step        
        """
        self.past_stages.clear()
        self.past_positions.clear()
        self.past_shifts.clear()
        self.past_Is.clear()

        if self.stage in self.stages:
            istage = list(self.stages.keys()).index(self.stage)
        else:
            istage = 0
        
        self._istages = iter(list(self.stages.keys())[istage:])
        self.stage = next(self._istages)

        while self.stage != 'done':
            p, dxdydz, I = self.next_step()
            
            yield self.stage, p, dxdydz, I
            
            self._prepare_next(p, dxdydz, I)
        
        self.stage = None

    def plot(self):
        """
        create a plot of the autofocussing process.

        This needs to be updated each iteration with `autofocus.update`.
        """

        self.fig = plt.figure()
        
        self.ax_3d = self.fig.add_subplot(211, projection='3d')
        self.I_ax = self.fig.add_subplot(212)
        
        self.fig.show()

        return self.fig

    def update(self, stage, p, dxdydz, I):
        """
        Update the autofocus plot with the latest data
        """
        if self.past_positions:
            xs, ys, zs = zip(*(self.past_positions))
        else:
            xs, ys, zs = [p[0]], [p[1]], [p[2]]

        self.ax_3d.clear()
        self.ax_3d.set_title(f'{stage} I={I:.3e}')
        self.ax_3d.scatter(xs, ys, zs, c=self._past_Is)
        self.ax_3d.plot(xs, ys, zs=zs)

        self.I_ax.clear()
        self.I_ax.plot(self.past_Is, '-o')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def do_autofocus(
        probe_ctrl: ProbeCtrlABC, 
        pump_ctrl: PumpCtrlABC, 
        get_response: ResponseFunc,
        width=1.0,
        rate: float | Sequence[float]=1e2,
        min_thresh=1e-5,
        min_step=0.05,
        dr=0.05,
        flat_thresh=5e-2,
        first_stage: None | Literal[
            'coarse-initial',
            'coarse-xy',
            'fine-xy',
            'central-xy',
            'coarse-focus',
            'fine-focus',
            'central-focus',
        ] = None,
    ) -> Generator[tuple[str, tuple[float, float, float], tuple[float, float, float], float], None, None]:
    """
    Perform an autofocus measurement.

    This function is to be used in for loops, see example below.

    This is basically just a function wrapper over the AutoFocus object.

    Arguments
    ---------
    probe_ctrl: ProbeCtrlABC
        The object autofocus uses to control the probe. It should have the methods described in `ProbeCtrlABC`.
    pump_ctrl: PumpCtrlABC
        The object autofocus uses to control the pump. It should have the methods described in `PumpCtrlABC`.
    get_response: ResponseFunc
        The function called to get the signal/ response at a given time. This is the parameter autofocus is trying to maximize.
    width: float
        This is the approx. width of the beam. *In practice this is used to set the maximum step size*. The algorithm will never
        step more than this value to try to increase the signal. This ensures it doesn't get over-excited and step over the optimal focus.
    rate: float, [float, float, float]
        The step size is calculated by multiplying the gradient by the rate, along with coarsening factors. The rate is stored as a numpy array
        of length 3. Where each float corresponds to [probe x, probe y, focus z]. If a single rate value is provided, this is used for
        each dimension i.e. set to [rate, rate, rate].
    min_thresh: float
        The autofocus has coarse stages for xy and focus which take larger steps vs the gradient dr. The autofocus will not move on from coarse
        stages, unless the signal is over the min_thresh.
    min_step: float
        In the fine xy/focus stages, as you get close to the best alignment (and also due to a decaying feature in `self.rate`) the step sizes
        will get quite small, and so this is a sign you are close to well focussed. min_step is the value at which autofocus considers the 
        step sufficiently small, it thinks it's close to best focus, and so will move on the central xy/ focus stage.
    dr: float
        dr is the amount the scanning mirror and xyz stage move when calculating the gradient. Because the alignment sensitivity is less for 
        the z direction, the dr in z is actually multiplied by `Autofocus.z_dr_multiplier`, which by default is 5...
    flat_thresh: float
        In the central focus steps, the autofocus will look back at central_lookback number of steps, and if the intensities are within the flat_thresh
        amount, the autofocus will consider the intenisty constant, and considers itself well focussed.
    first_stage: str
        This string represents which stage we are in from `Autofocus.stages`. You can explicitly set this value to a stage and the autofocus will proceed 
        from this stage.

    Yields
    ------
    stage: str
        The current stage
    p: tuple[float, float, float]
        The position after the step
    dxdydz: tuple[float, float, float]
        The last movement to improve the focus
    I: float
        The intensity after the step

    Example
    -------
    ```python
    from ustm.drivers import (
        Control_LI, Control_ScanningMirror, Control_xyz_stage
    )

    from ustm.autofocus import AutoFocus, ProbeCtrl, PumpCtrl, wrap_lockin

    # Just as in the ST_GUIs, you need to first connect to the devices you need using a with
    # statement
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
    """
    return AutoFocus(
        probe_ctrl=probe_ctrl,
        pump_ctrl=pump_ctrl,
        get_response=get_response,
        width=width,
        rate=rate,
        min_thresh=min_thresh,
        min_step=min_step,
        dr=dr,
        flat_thresh=flat_thresh,
        first_stage=first_stage
    )
