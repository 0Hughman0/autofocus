import json
from unittest.mock import Mock
from typing import TypeVar
from typing_extensions import Iterable

import numpy as np
import matplotlib.pyplot as plt
import pytest

from autofocus.test_utils import VLaser
from autofocus import AutoFocus, do_autofocus
from autofocus.autofocus import (
    coarse_xy_done, 
    fine_xy_done, 
    central_xy_done,
    coarse_focus_done,
    fine_focus_done,
    central_focus_done,
    central_focus_I_done
)
from autofocus.test_utils import VFocus, VProbeMirror, VLockin, TAutoFocus


def factory(ix=0, iy=0, iz=0, noise=0):
    """
    ix: the x of the scanning mirror
    iy: the y of the scanning mirror
    iz: the z of the scanning stage i.e. focus

    Will always set pump and probe lasers to be best aligned at 0, 0, 0.
    """
    pump = VLaser(x0=0, y0=0, z0=0)
    probe = VLaser(x0=0, y0=0, z0=0)

    probe_ctrl = VProbeMirror(probe, ix=ix, iy=iy)
    pump_ctrl = VFocus(iz=iz)

    vlockin = VLockin(
        probe=probe,
        pump=pump,
        focus=pump_ctrl,
        probe_mirror=probe_ctrl,
        noise=noise
    )

    get_reponse = Mock(side_effect=lambda x, y, z: vlockin.poll())

    auto = TAutoFocus(
        probe_ctrl=probe_ctrl,
        pump_ctrl=pump_ctrl,
        get_response=get_reponse,
        probe=probe,
        pump=pump,
        lockin=vlockin
    )

    auto.shift = Mock(side_effect=auto.shift)
    
    return auto


@pytest.fixture
def mk_autofocus():
    return factory


def test_construct(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)

    assert isinstance(auto.rate, np.ndarray)
    assert isinstance(auto.dr, np.ndarray)
    assert auto.dr[-1] == auto.dr[0] * auto.z_dr_multiplier
    

def test_next_dr(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.stage = None
    dr = auto._next_dr()
    assert np.all(dr == auto.dr)


def test_next_dr_flipflops(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)

    dr = auto._next_dr()
    flip_dr = auto._next_dr()
    assert np.all(dr == -flip_dr)


def test_coarse_dr(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    
    auto.stage = 'coarse'
    auto.dr_coarsening_factor = 17

    dr = auto._next_dr()
    assert np.all(dr == auto.dr * auto.dr_coarsening_factor)


def test_next_dr_xy_no_z(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    
    auto.stage = 'coarse-xy'

    assert np.all(auto._next_dr()[:2] != 0)
    assert auto._next_dr()[2] == 0
    


def test_next_dr_focus_no_xy(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)

    auto.stage = 'coarse-focus'

    assert np.all(auto._next_dr()[:2] == 0)
    assert auto._next_dr()[2] != 0


def test_grad_shifts(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    
    auto.stage = None
    shift_calls = auto.shift.call_args_list

    dr = auto.dr

    auto._grad()

    for i, expect in enumerate([
        (dr[0], 0, 0),
        (0, dr[1], 0),
        (0, 0, dr[2])
    ]):
        assert shift_calls[i].args == expect

    shift_calls.clear()
    auto._grad()

    for i, expect in enumerate([
        (-dr[0], 0, 0),  # the flip
        (0, -dr[1], 0),
        (0, 0, -dr[2]),
    ]):        
        assert shift_calls[i].args == expect
        


def test_grad_response(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.stage = None

    response_calls = auto.get_response.call_args_list

    dr = auto.dr
    auto._grad()
    
    for i, expect in enumerate([
        (0, 0, 0),
        (dr[0], 0, 0),
        (0, dr[1], 0),
        (0, 0, dr[2]),
    ]):
        assert response_calls[i].args == expect 


def test_grad_calc(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)

    auto.get_response.side_effect = [
        1, # for 0, 0, 0
        2, # for dr, 0, 0
        3, # for 0, dr, 0
        4, # for 0, 0, dr
    ]

    auto.stage = None
    dr = auto.dr
    grad = auto._grad()

    assert auto.shift.call_count == 3

    assert grad[0] == pytest.approx((2 - 1) / dr[0])
    assert grad[1] == pytest.approx((3 - 1) / dr[1])
    assert grad[2] == pytest.approx((4 - 1) / dr[2])



def test_grad_calc_xy(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.stage = 'xy'

    auto.get_response.side_effect = [
        1, # for 0, 0, 0
        2, # for dr, 0, 0
        3, # for 0, dr, 0
    ]

    dr = auto.dr
    grad = auto._grad()

    assert auto.shift.call_count == 2

    assert grad[0] == pytest.approx((2 - 1) / dr[0])
    assert grad[1] == pytest.approx((3 - 1) / dr[1])
    assert grad[2] == 0


def test_grad_calc_focus(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.stage = 'focus'

    auto.get_response.side_effect = [
        1, # for 0, 0, 0
        4 # for 0, 0, dr
    ]

    dr = auto.dr
    grad = auto._grad()

    assert auto.shift.call_count == 1

    assert grad[0] == 0
    assert grad[1] == 0
    assert grad[2] == pytest.approx((4 - 1) / dr[2])


def test_xy_improves(mk_autofocus):
    auto: TAutoFocus = mk_autofocus(ix=-0.5, iy=0, iz=0)
    I0 = auto.poll()
    x0, *_ = auto.positions()

    auto.next_step()
    I1 = auto.poll()
    x1, *_ = auto.positions()
    assert I1 > I0
    assert abs(x1) < abs(x0)  # closer to aligned

    auto = mk_autofocus(ix=0, iy=0.5, iz=0)
    I0 = auto.poll()
    _, y0, _ = auto.positions()

    auto.next_step()
    I1 = auto.poll()
    _, y1, _ = auto.positions()
    assert I1 > I0
    assert abs(y1) < abs(y0)


def test_xy_z_unchanged(mk_autofocus):
    auto = mk_autofocus(ix=-0.5, iy=0, iz=-1)
    _, _, z0 = auto.positions()

    auto.stage = 'xy'
    auto.next_step()

    _, _, z1 = auto.positions()

    assert z0 == z1


def test_overshoot_detection(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)

    old_rate = auto.rate

    auto._prepare_next((0, 0, 0), np.array([-2, 2, -2]), 1)
    auto._prepare_next((0, 0, 0), np.array([1, 1, 1]), 1)

    assert auto.rate[0] == old_rate[0] * 0.75
    assert auto.rate[1] == old_rate[1]  # y did not flip
    assert auto.rate[2] == old_rate[2] * 0.75

@pytest.mark.parametrize(
        'past_stages,expect',
        [
            [[], None],
            [['a'] * 1, 1],
            [['a'] * 2, 2],
            [['a'] * 3, 3],
            [['a'] * 4, 4],
            [['a'] * 5, 4],
            [['a', 'a', 'b', 'a', 'a'], 2]
        ]        
)
def test_n_recent(mk_autofocus, past_stages, expect):
    auto: TAutoFocus = mk_autofocus(ix=0, iy=0, iz=0)
    auto.central_lookback = 4
    auto.stage = 'a'
    auto._past_stages = past_stages

    index = auto._n_recent()

    assert index == expect

    if index is not None:
        assert len(list(range(10))[-auto._n_recent():]) == expect


@pytest.mark.parametrize(
        'I,expect',
        [
            [0, False],
            [0.75, True],
        ]
)
def test_coarse_xy_done(mk_autofocus, I, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    
    assert coarse_xy_done(auto, (0, 0, 0), (0, 0, 0), I) == expect


@pytest.mark.parametrize(
        'dxdydz,I,expect',
        [
            [(0, 0, 0), 0, False],
            [(0, 0, 0), 0.75, False],
            [(0, 0, 0), 1.5, True],
            [(2, 2, 0), 1.5, False],
            [(2, 2, 0), 0.75, False],
        ]
)
def test_fine_xy_done(mk_autofocus, dxdydz, I, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    
    assert fine_xy_done(auto, (0, 0, 0), dxdydz, I) == expect


@pytest.mark.parametrize(
        'past_stages,past_shifts,expect',
        [
            [
                ['b', 'b', 'b', 'a'], 
                [
                    (0.5, 0.5, 0),
                    (-0.5, -0.5, 0),
                    (0.5, 0.5, 0)
                ], 
                False
            ],
            [
                ['a', 'a', 'a', 'a'], 
                [
                    (-0.5, -0.5, 0), 
                    (0.5, 0.5, 0),
                    (-0.5, -0.5, 0),
                    (0.5, 0.5, 0)
                ],
                True
            ],
            [
                ['a', 'a', 'a', 'a'],
                [
                    (-0.5, -0.5, 0), 
                    (0.5, 0.5, 0),
                    (0.5, -0.5, 0),  # x did not flip.
                    (0.5, 0.5, 0)
                ],
                False
            ],
            [
                ['a', 'a', 'a', 'a'],  # p x did not flip
                [
                    (-0.5, -0.5, 0), 
                    (0.5, 0.5, 0),
                    (-0.5, -0.5, 0),
                    (-0.5, 0.5, 0)
                ],
                False
            ],
            [
                ['a', 'a', 'a', 'a'],
                [
                    (-0.5, -0.5, 0),  # ignores old, non-flippers!
                    (-0.5, -0.5, 0), 
                    (0.5, 0.5, 0),
                    (-0.5, -0.5, 0),
                    (0.5, 0.5, 0)
                ],
                True
            ],
        ]
)
def test_central_xy_done(mk_autofocus, past_stages, past_shifts, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    auto.stage = 'a'
    auto._past_stages = past_stages
    auto._past_shifts = past_shifts
    auto._past_positions = [[0, 0, 0]]
    auto.central_lookback = 4
    
    assert central_xy_done(auto, (0, 0, 0), (0, 0, 0), 0) == expect



@pytest.mark.parametrize(
        'dxdydz,I,expect',
        [
            [(0, 0, 0), 0, False],
            [(0, 0, 0), 0.75, False],
            [(0, 0, 0), 1.5, True],
            [(0, 0, 2), 1.5, False],
            [(0, 0, 2), 0.75, False],
        ]
)
def test_fine_focus_done(mk_autofocus, dxdydz, I, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    
    assert fine_focus_done(auto, (0, 0, 0), dxdydz, I) == expect


@pytest.mark.parametrize(
        'I,expect',
        [
            [0, False],
            [0.75, False],
            [1.5, True],
        ]
)
def test_coarse_focus_done(mk_autofocus, I, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    auto._past_positions = [[]]
    
    assert coarse_focus_done(auto, (0, 0, 0), (0, 0, 0), I) == expect



@pytest.mark.parametrize(
        'past_stages,past_shifts,expect',
        [
            [['b', 'b', 'b', 'a'], [
                (0, 0, -0.5), 
                (0, 0, 0.5),
                (0, 0, -0.5),
                (0, 0, 0.5),], False],
            [['a'] * 4, [
                (0, 0, -0.5), 
                (0, 0, 0.5),
                (0, 0, -0.5),
                (0, 0, 0.5),
                ],
                True
            ],
            [['a'] * 4, [
                (0, 0, -0.5), 
                (0, 0, 0.5),
                (0, 0, 0.5),  # z did not flip.
                (0, 0, 0.5)
                ],
                False
            ],
            [['a'] * 4, [
                (0, 0, -0.5), 
                (0, 0, 0.5),
                (0, 0, -0.5),
                (0, 0, -0.5) # final dz did not flip
                ],
                False
            ],
            [['a'] * 4, [
                (0, 0, -0.5),  # ignores old, non-flippers!
                (0, 0, -0.5), 
                (0, 0, 0.5),
                (0, 0, -0.5),
                (0, 0, 0.5)
                ],
                True
            ],
        ]
)
def test_central_focus_done(mk_autofocus, past_stages, past_shifts, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    auto.stage = 'a'
    auto._past_stages = past_stages
    auto._past_shifts = past_shifts
    auto._past_positions = [[0, 0, 0]]
    auto.central_lookback = 4
    
    assert central_focus_done(auto, (0, 0, 0), (0, 0, 0), 0) == expect


@pytest.mark.parametrize(
        'past_stages,past_Is,expect',
        [
            [['b', 'b', 'b', 'a'], [1, 1, 1, 1], False],  # can't be too short history
            [['a'] * 4, [1, 1, 1, 1], True],
            [['a'] * 4, [3, 1, 1, 1], False], 
            [['a'] * 4, [1, 1, 1, 3], False],
            [['a'] * 4, [3, 3, 3, 1], False]
        ]
)
def test_central_focus_I_done(mk_autofocus, past_stages, past_Is, expect):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.min_step = 1
    auto.min_thresh = 1
    auto.flat_thresh = 0.5
    
    auto.stage = 'a'
    auto._past_stages = past_stages
    auto._past_Is = past_Is
    auto._past_positions = [[0, 0, 0]]
    auto.central_lookback = 4
    
    assert central_focus_I_done(auto, (0, 0, 0), (0, 0, 0), 0) == expect


def test_central_xy_average_position(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.go = Mock(side_effect=auto.go)

    auto.central_lookback = 4

    auto.stage = 'a'
    auto._past_stages = ['a'] * 4
    auto._past_shifts = [
        [-1, -1, 0], [1, 1, 0], [-1, -1, 0], [1, 1, 0], 
    ]
    auto._past_positions = [
        (-3, -10, 0),
        (1, 5, 0),
        (-3, -10, 0),
        (1, 5, 0),
    ]

    central_xy_done(auto, None, None, None)

    assert list(auto.go.call_args.args) == [-1.0, -2.5, None]


def test_central_focus_average_position(mk_autofocus):
    auto = mk_autofocus(ix=0, iy=0, iz=0)
    auto.go = Mock(side_effect=auto.go)

    auto.central_lookback = 4
    auto.stage = 'a'
    auto._past_stages = ['a'] * 4
    auto._past_shifts = [
        [0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, -1]
    ]
    auto._past_positions = [
        (0, 0, -2),
        (0, 0, 3),
        (0, 0, -2),
        (0, 0, 3),
    ]

    central_focus_done(auto, None, None, None)

    assert list(auto.go.call_args.args) == [None, None, 0.5]


def test_iterating_through(mk_autofocus):
    auto: TAutoFocus = factory(ix=0, iy=0.0, iz=0.0, noise=0)

    once = Mock(return_value=True)
    twice = Mock(side_effect=[False, True])
    
    auto.stages = {
        '1 once': once,
        '2 twice': twice,
        '3 once': once,
    }

    iauto = iter(auto)

    stage, *_ = next(iauto)
    assert stage == '1 once'
    stage, *_ = next(iauto)
    assert stage == '2 twice'
    stage, *_ = next(iauto)
    assert stage == '2 twice'
    stage, *_ = next(iauto)
    assert stage == '3 once'
    

def test_focus_improves(mk_autofocus):
    auto: TAutoFocus = mk_autofocus(ix=0, iy=0, iz=-1)

    I0 = auto.poll()
    _, _, z0 = auto.positions()

    auto.stage = 'focus'
    auto.next_step()

    I1 = auto.poll()
    _, _, z1 = auto.positions()

    assert I1 > I0
    assert abs(z1) < abs(z0)  # focus improved


def main():
    auto: TAutoFocus = factory(ix=-1, iy=0.5, iz=-0.5, noise=0.1)

    print(auto.poll())
    print(auto._grad())

    camera = auto.plot_camera()
    auto.plot()

    try:
        for i, step in enumerate(auto):
            print(step)
        
            camera.update()
            auto.update(*step[1:])
    except Exception as e:
        print(e)
    finally:
        return auto


if __name__ == '__main__':
    auto = main()