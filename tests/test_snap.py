import numpy as np

from mtah.common import A, D, H
from mtah.snap import snap, touches


def scene(T=200, kick=100):
    """Ball rolls slowly, then is kicked at `kick` by the away player standing next to it."""
    bxy = np.zeros((T, 2))
    bxy[kick:, 0] = np.arange(T - kick) * 0.4              # 10 m/s after the kick
    pxy = np.full((T, 2, 2), 30.0)
    pxy[:, 0] = (-1.0, 0.0)                                # home player, 1 m away from the ball
    pxy[:kick, 1] = (1.0, 0.5)                             # away player 1.1 m away, then gone
    return pxy, np.array([0, 1], dtype=np.int8), bxy


def test_touch_found_with_the_team_of_the_nearest_player():
    pxy, team, bxy = scene()
    frames, side = touches(pxy, team, bxy)
    assert len(frames) == 1 and abs(frames[0] - 100) <= 3
    assert side[0] == 0                                    # home player is nearer (1.0 m < 1.1 m)


def test_snap_moves_a_team_switch_to_the_touch_and_respects_min_dur():
    frames, side = np.array([100]), np.array([A], dtype=np.int8)
    codes = np.array([H] * 90 + [A] * 110, dtype=np.int8)  # decoder put the switch at 90
    out = snap(codes, [(0, 200)], frames, side, (25, 25, 50), W=25)
    assert (out[:100] == H).all() and (out[100:] == A).all()
    # too far away: untouched
    assert (snap(codes, [(0, 200)], frames, side, (25, 25, 50), W=5) == codes).all()
    # would leave the first segment shorter than min_dur: untouched
    short = np.array([H] * 30 + [A] * 170, dtype=np.int8)
    assert (snap(short, [(0, 200)], np.array([20]), side, (25, 25, 50), W=25) == short).all()


def test_snap_restart_and_wrong_team_touch():
    codes = np.array([D] * 60 + [A] * 140, dtype=np.int8)
    out = snap(codes, [(0, 200)], np.array([70]), np.array([A], dtype=np.int8), (25, 25, 50), W=25)
    assert (out[:70] == D).all() and (out[70:] == A).all()
    # a touch by the other team does not move the boundary
    out = snap(codes, [(0, 200)], np.array([70]), np.array([H], dtype=np.int8), (25, 25, 50), W=25)
    assert (out == codes).all()
