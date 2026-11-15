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
from numba import njit

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
    ball_far_boost: float = 1.0     # drop hazard x this when no player within 4 m (ball in flight)
    ball_out_boost: float = 1.0     # drop hazard x this when the ball is outside the pitch
    fov_missing: float = 0.0        # fraction of frames where one end of the pitch is out of view
    fov_burst: float = 150.0        # mean length of such a stretch, frames
    false_ball: float = 0.0         # fraction of frames with a false ball on a random player
    fragment: bool = False          # cut tracks into tracklets packed into reused slots
    vball_p: float = 0.0            # prob. per sample of refilling ball gaps like the video ball
                                    # detector would (vball_fill, params from vball_params)
    flip: bool = True
    team_swap: bool = True
    enabled: bool = True

    def to_dict(self):
        return asdict(self)


@njit(cache=True)
def _markov(u, state, p_up, hazard):
    T, n = u.shape
    out = np.empty((T, n), dtype=np.bool_)
    for t in range(T):
        for j in range(n):
            if state[j]:
                state[j] = u[t, j] >= p_up
            else:
                state[j] = u[t, j] < hazard[t, j]
            out[t, j] = state[j]
    return out


def markov_missing(rng, T, n, missing, burst, boost=None):
    """(T, n) bool, True = missing. Two-state Markov chain per column with bursty dropouts.

    `boost` (T, n) multiplies the drop hazard frame by frame (e.g. crowded ball, ball in the air).
    Draws exactly the same random numbers as the original pure-Python loop (u, then the initial
    state), so fixed-seed validation sets are unchanged.
    """
    if missing <= 0:
        return np.zeros((T, n), dtype=bool)
    p_up = 1.0 / max(burst, 1.0)
    p_down = min(0.99, missing / max(1e-6, 1.0 - missing) * p_up)
    u = rng.random((T, n))
    state = rng.random(n) < missing
    hazard = np.full((T, n), p_down) if boost is None else         np.ascontiguousarray(np.broadcast_to(np.minimum(0.99, p_down * boost), (T, n)), dtype=np.float64)
    return _markov(u, state.copy(), p_up, hazard)


def randomized_v1(cfg, rng):
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


def randomized(cfg, rng):
    """Domain randomisation v2, centred on what mtah.noise_stats measured on the real ATD:
    ball gaps of ~10-30 s (not 1 s), ball lost more when far from players, whole ends of the pitch
    out of view at times, very few false balls, smoother ball positions. Short-gap regimes are
    kept in the mix so nothing learned in v1 is unlearned."""
    long_gaps = rng.random() < 0.7
    return replace(
        cfg,
        ball_missing=rng.uniform(0.18, 0.55),
        ball_burst=float(np.exp(rng.uniform(np.log(150), np.log(900)))) if long_gaps
        else rng.uniform(20, 100),
        ball_crowd_boost=rng.uniform(0.5, 2.0), ball_far_boost=rng.uniform(1.0, 3.0),
        ball_out_boost=rng.uniform(1.0, 10.0),
        ball_jitter=rng.uniform(0.05, 0.5),
        player_missing=rng.uniform(0.03, 0.15), player_burst=rng.uniform(30, 300),
        player_jitter=rng.uniform(0.1, 0.4),
        fov_missing=rng.uniform(0.0, 0.15), fov_burst=rng.uniform(50, 300),
        false_ball=rng.uniform(0.0, 0.02) if rng.random() < 0.25 else 0.0,
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


def vball_fill(bxy, clean_bxy, pxy, rng, fill, prec, err, burst=1.0):
    """Simulate refilling ball gaps with a video ball detector (Jalur 1).

    Each frame where the corrupted ball is missing (and the clean ball exists) gets a detection with
    prob `fill`; that detection is right with prob `prec` (clean ball + N(0, err) m) and otherwise
    lands on a random visible player (+0.5 m noise) -- the worst place for possession. Frames are
    independent, like a per-frame detector. Uses its own rng, so the underlying scenario is
    unchanged.

    burst > 1: wrong detections come in bursts (Markov, mean `burst` frames) that stay on ONE
    player for the whole burst -- a persistent look-alike, much harder to average out.
    """
    bxy = bxy.copy()
    gap = np.flatnonzero(np.isnan(bxy[:, 0]) & ~np.isnan(clean_bxy[:, 0]))
    hit = gap[rng.random(len(gap)) < fill]
    if burst <= 1:
        right = rng.random(len(hit)) < prec
        r = hit[right]
        bxy[r] = clean_bxy[r] + rng.normal(0, err, (len(r), 2)).astype(np.float32)
        for t in hit[~right]:
            vis = np.flatnonzero(~np.isnan(pxy[t, :, 0]))
            if len(vis):
                bxy[t] = pxy[t, rng.choice(vis)] + rng.normal(0, 0.5, 2).astype(np.float32)
        return bxy
    T = len(bxy)
    wrong = markov_missing(rng, T, 1, 1.0 - prec, burst)[:, 0]
    is_hit = np.zeros(T, bool)
    is_hit[hit] = True
    r = np.flatnonzero(is_hit & ~wrong)
    bxy[r] = clean_bxy[r] + rng.normal(0, err, (len(r), 2)).astype(np.float32)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], wrong.astype(np.int8), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        vis = np.flatnonzero(~np.isnan(pxy[a, :, 0]))
        if not len(vis):
            continue
        s = rng.choice(vis)
        off = rng.normal(0, 0.5, 2).astype(np.float32)
        t = np.flatnonzero(is_hit[a:b]) + a
        bxy[t] = pxy[t, s] + off           # NaN where that player is not tracked: no detection
    return bxy


def vball_params(rng):
    """Random video-ball detector for training (run5, WORKLOG 2026-11-16). Ranges are wide around
    what ft1 measured (fill 0.54, precision 0.74 vs a noisy reference, error 0.6 m); not tuned."""
    return dict(fill=rng.uniform(0.3, 0.8), prec=rng.uniform(0.5, 0.95), err=rng.uniform(0.2, 1.0),
                burst=1.0 if rng.random() < 0.3 else rng.uniform(5.0, 30.0))


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
    clean_ball = ball_xy
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
    far = np.nan_to_num(d, nan=1e9).min(axis=1) > 4.0
    out = (np.abs(np.nan_to_num(ball_xy[:, 0])) > 52.5) | (np.abs(np.nan_to_num(ball_xy[:, 1])) > 34.0)
    boost = (np.where(crowd, cfg.ball_crowd_boost, 1.0) * np.where(far, cfg.ball_far_boost, 1.0)
             * np.where(out, cfg.ball_out_boost, 1.0))
    if ball_z is not None:
        boost = boost * np.where(np.nan_to_num(ball_z.reshape(-1)) > 2.0, cfg.ball_air_boost, 1.0)
    gone = markov_missing(rng, T, 1, cfg.ball_missing, cfg.ball_burst, boost[:, None])[:, 0]
    ball_xy += rng.normal(0, cfg.ball_jitter, ball_xy.shape).astype(np.float32)
    ball_xy[gone] = np.nan

    # players: bursty dropouts per slot, jitter
    pgone = markov_missing(rng, T, S, cfg.player_missing, cfg.player_burst)
    players_xy += rng.normal(0, cfg.player_jitter, players_xy.shape).astype(np.float32)
    players_xy[pgone] = np.nan

    if cfg.fov_missing > 0:   # the camera does not see one end: everyone beyond a line vanishes
        off = markov_missing(rng, T, 1, cfg.fov_missing, cfg.fov_burst)[:, 0]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], off.astype(np.int8), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            side, cut = rng.choice([-1.0, 1.0]), rng.uniform(10, 40)
            hide = side * players_xy[a:b, :, 0] > cut
            players_xy[a:b][hide] = np.nan

    if cfg.false_ball > 0:
        ball_xy = add_false_ball(ball_xy, players_xy, rng, cfg.false_ball)
    if cfg.fragment:
        players_xy, team = fragment_tracks(players_xy, team, rng)
    # same place as in mtah.stress.scenario_inputs: after the tracking noise, before the flips.
    # No rng is drawn when vball_p == 0, so older configs corrupt exactly as before.
    if cfg.vball_p > 0 and rng.random() < cfg.vball_p:
        ball_xy = vball_fill(ball_xy, clean_ball, players_xy, rng, **vball_params(rng))

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
