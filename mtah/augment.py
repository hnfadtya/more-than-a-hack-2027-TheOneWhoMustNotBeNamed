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

from dataclasses import asdict, dataclass, replace

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
    false_ball: float = 0.0         # fraction of frames with a false ball on a random player
    fragment: bool = False          # cut tracks into tracklets packed into reused slots
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


def randomized(cfg, rng):
    """Domain randomisation: a fresh corruption level per sample, wide enough to cover whatever the
    real ATD turns out to be (stress test: the net was weakest on false balls and noisy players)."""
    return replace(
        cfg,
        ball_missing=rng.uniform(0.30, 0.60), ball_burst=rng.uniform(20, 100),
        ball_jitter=rng.uniform(0.2, 1.0),
        player_missing=rng.uniform(0.03, 0.20), player_burst=rng.uniform(30, 120),
        player_jitter=rng.uniform(0.1, 0.6),
        false_ball=rng.uniform(0.0, 0.06) if rng.random() < 0.5 else 0.0,
        fragment=bool(rng.random() < 0.5))


def add_false_ball(ball_xy, players_xy, rng, rate, burst=12.0):
    """Bursts where the 'ball' sits on a random player (the classic false positive)."""
    T = len(ball_xy)
    fake = markov_missing(rng, T, 1, rate, burst)[:, 0]
    out = ball_xy.copy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], fake.astype(np.int8), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        valid = np.flatnonzero(~np.isnan(players_xy[a, :, 0]))
        if len(valid):
            s = rng.choice(valid)
            out[a:b] = players_xy[a:b, s] + rng.normal(0, 0.5, 2)
    return out


def fragment_tracks(players_xy, team, rng, mean_len=1500, wrong_team=0.03, dup=0.03):
    """Cut every track into tracklets and pack them into reused slots, ATD style.

    Each tracklet goes to a free slot of its team -- or, with prob `wrong_team`, of the other team
    (a jersey-colour mistake). With prob `dup` a tracklet is also copied, slightly offset, into a
    second slot (the tracker following one person twice).
    """
    T, S, _ = players_xy.shape
    segs = []
    for s in range(S):
        present = ~np.isnan(players_xy[:, s, 0])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], present.astype(np.int8), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            cuts = np.sort(rng.integers(a, b, size=rng.poisson((b - a) / mean_len)))
            bounds = np.concatenate(([a], cuts, [b]))
            for x, y in zip(bounds[:-1], bounds[1:]):
                if y > x:
                    t = team[s] if rng.random() >= wrong_team else 1 - team[s]
                    segs.append((x, y, s, t, 0.0))
                    if rng.random() < dup:
                        segs.append((x, y, s, t, 0.7))
    segs.sort()
    slots = {0: [], 1: []}          # per team: list of [free_at, slot index]
    new_team, assign = [], []
    for x, y, s, t, off in segs:
        free = [k for k, (f, _) in enumerate(slots[t]) if f <= x]
        if free:
            k = free[0]
            idx = slots[t][k][1]
            slots[t][k][0] = y
        else:
            idx = len(new_team)
            new_team.append(t)
            slots[t].append([y, idx])
        assign.append((x, y, s, idx, off))
    out = np.full((T, max(1, len(new_team)), 2), np.nan, dtype=np.float32)
    for x, y, s, idx, off in assign:
        out[x:y, idx] = players_xy[x:y, s] + (rng.normal(0, off, 2) if off else 0)
    return out, np.array(new_team or [0], dtype=np.int8)


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

    if cfg.false_ball > 0:
        ball_xy = add_false_ball(ball_xy, players_xy, rng, cfg.false_ball)
    if cfg.fragment:
        players_xy, team = fragment_tracks(players_xy, team, rng)

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
