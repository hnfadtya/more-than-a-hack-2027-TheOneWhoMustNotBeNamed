"""Make clean tracking look like the provided ATD, so a model trained on other games transfers.

External tracking (IDSSE/DFL) is corrected: every player on every frame, the ball always there.
The ATD is not: the ball is missing on ~41% of graded frames, in bursts, and disproportionately
when it is crowded or in the air; tracks drop out and fragment. A model that never saw that would
lean on the ball being present and fall over on day one.

The defaults below are guesses. Once the competition data arrives, measure the ATD's own statistics
(missing fraction, burst lengths, tracked-object counts) with `mtah.atd` and set `CorruptCfg` from
them -- measuring the input is fine, it uses no labels.

Also: random pitch flips and a home/away swap (labels swapped with it), so the model learns
possession from geometry, not from which side is which.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .common import A, H


@dataclass
class CorruptCfg:
    ball_missing: float = 0.41      # target stationary missing fraction (before boosts)
    ball_burst: float = 40.0        # mean missing burst length, frames
    ball_crowd_boost: float = 2.0   # drop hazard x this when >= 3 players within 3 m of the ball
    ball_air_boost: float = 3.0     # drop hazard x this when ball z > 2 m (only if z is known)
    ball_jitter: float = 0.3        # metres, gaussian
    player_missing: float = 0.07    # ~20.5 of 22 tracked during play
    player_burst: float = 50.0
    player_jitter: float = 0.15
    flip: bool = True
    team_swap: bool = True
    enabled: bool = True

    def to_dict(self):
        return asdict(self)


def markov_missing(rng, T, n, missing, burst, boost=None):
    """(T, n) bool, True = missing. Two-state Markov chain per column with bursty dropouts.

    `boost` (T, n) multiplies the drop hazard frame by frame (e.g. crowded ball, ball in the air).
    """
    if missing <= 0:
        return np.zeros((T, n), dtype=bool)
    p_up = 1.0 / max(burst, 1.0)
    p_down = min(0.99, missing / max(1e-6, 1.0 - missing) * p_up)
    u = rng.random((T, n))
    out = np.empty((T, n), dtype=bool)
    state = rng.random(n) < missing
    for t in range(T):
        hazard = p_down if boost is None else np.minimum(0.99, p_down * boost[t])
        state = np.where(state, u[t] >= p_up, u[t] < hazard)
        out[t] = state
    return out


def corrupt(players_xy, team, ball_xy, label=None, rng=None, cfg=CorruptCfg(), ball_z=None):
    """Return corrupted copies (players_xy, team, ball_xy, label). Inputs are never modified."""
    rng = rng or np.random.default_rng()
    players_xy = players_xy.copy()
    ball_xy = ball_xy.copy()
    team = team.copy()
    label = None if label is None else label.copy()
    if not cfg.enabled:
        return players_xy, team, ball_xy, label
    T, S, _ = players_xy.shape

    # ball: drop hazard rises when crowded or airborne
    d = np.linalg.norm(players_xy - ball_xy[:, None], axis=2)
    crowd = (np.nan_to_num(d, nan=1e9) < 3.0).sum(axis=1) >= 3
    boost = np.where(crowd, cfg.ball_crowd_boost, 1.0)
    if ball_z is not None:
        boost = boost * np.where(np.nan_to_num(ball_z.reshape(-1)) > 2.0, cfg.ball_air_boost, 1.0)
    gone = markov_missing(rng, T, 1, cfg.ball_missing, cfg.ball_burst, boost[:, None])[:, 0]
    ball_xy += rng.normal(0, cfg.ball_jitter, ball_xy.shape).astype(np.float32)
    ball_xy[gone] = np.nan

    # players: bursty dropouts per slot, jitter
    pgone = markov_missing(rng, T, S, cfg.player_missing, cfg.player_burst)
    players_xy += rng.normal(0, cfg.player_jitter, players_xy.shape).astype(np.float32)
    players_xy[pgone] = np.nan

    # geometry augmentations
    if cfg.flip:
        sign = np.array([rng.choice([-1, 1]), rng.choice([-1, 1])], dtype=np.float32)
        players_xy *= sign
        ball_xy *= sign
    if cfg.team_swap and rng.random() < 0.5:
        team = (1 - team).astype(np.int8)
        if label is not None:
            h, a = label == H, label == A
            label[h], label[a] = A, H
    return players_xy, team, ball_xy, label
