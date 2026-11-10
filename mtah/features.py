"""Per-frame model input: an unordered set of player tokens plus one ball token.

The ATD has no player identities (46 slots for 22 players, swapping), so nothing here relies on a
slot meaning the same person over time. Each frame keeps the P players nearest to the ball (or to
where it was last seen, if it is missing now) and describes each one relative to the ball. The
model treats them as a set.

Player token (F_PLAYER):  x, y, vx, vy, team(+1 home / -1 away), dx, dy, dist to ball ref,
                          is_nearest_of_own_team
Ball token (F_BALL):      x, y, vx, vy, visible, frames since seen, ref x, ref y,
                          n_home within 5 m, n_away within 5 m

Velocities come from central differences within a slot, so an identity switch shows as a spike;
they are clipped. Missing values become 0 with an explicit mask/flag.
"""

import numpy as np

P_MAX = 24
F_PLAYER = 9
F_BALL = 10
HL, HW = 52.5, 34.0


def _velocity(pos, fps):
    v = np.full_like(pos, np.nan)
    if len(pos) >= 3:
        v[1:-1] = (pos[2:] - pos[:-2]) * (fps / 2.0)
    if len(pos) >= 2:
        v[0] = (pos[1] - pos[0]) * fps
        v[-1] = (pos[-1] - pos[-2]) * fps
    return v


def _ffill_ball(ball_xy):
    """(last known position (T,2), frames since seen (T,)). NaN/large before the first sighting."""
    T = len(ball_xy)
    seen = ~np.isnan(ball_xy[:, 0])
    idx = np.where(seen, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    last = np.where((idx >= 0)[:, None], ball_xy[np.maximum(idx, 0)], np.nan)
    since = np.where(idx >= 0, np.arange(T) - idx, 10 ** 6)
    return last, since


def featurize(players_xy, team, ball_xy, fps=25):
    """-> players (T,P,F_PLAYER) f32, pmask (T,P) bool, ball (T,F_BALL) f32."""
    T, S, _ = players_xy.shape
    valid = ~np.isnan(players_xy[..., 0])
    pv = np.clip(np.nan_to_num(_velocity(players_xy, fps)), -12, 12)
    bvis = ~np.isnan(ball_xy[:, 0])
    bv = np.clip(np.nan_to_num(_velocity(ball_xy, fps)), -40, 40)
    last, since = _ffill_ball(ball_xy)
    ref = np.nan_to_num(last)                                   # (T,2), 0 before first sighting

    rel = players_xy - ref[:, None]                              # (T,S,2)
    dist = np.linalg.norm(rel, axis=2)
    key = np.where(valid, dist, np.inf)
    order = np.argsort(key, axis=1, kind="stable")[:, :P_MAX]    # (T,P')
    P = order.shape[1]

    g = lambda a: np.take_along_axis(a, order, axis=1)          # noqa: E731
    gxy = np.take_along_axis(players_xy, order[..., None], axis=1)
    gv = np.take_along_axis(pv, order[..., None], axis=1)
    grel = np.take_along_axis(rel, order[..., None], axis=1)
    gdist = g(dist)
    gmask = g(valid)
    gteam = team[order]                                          # (T,P)
    sign = np.where(gteam == 0, 1.0, -1.0)

    # nearest of own team: first occurrence of each team in the distance-sorted list
    nearest = np.zeros((T, P), dtype=bool)
    for k in (0, 1):
        is_k = (gteam == k) & gmask
        first = np.argmax(is_k, axis=1)
        has = is_k.any(axis=1)
        nearest[np.flatnonzero(has), first[has]] = True

    players = np.zeros((T, P_MAX, F_PLAYER), dtype=np.float32)
    feats = np.stack([
        gxy[..., 0] / HL, gxy[..., 1] / HW, gv[..., 0] / 10, gv[..., 1] / 10, sign,
        grel[..., 0] / 20, grel[..., 1] / 20, gdist / 20, nearest.astype(np.float64),
    ], axis=-1)
    players[:, :P] = np.where(gmask[..., None], np.nan_to_num(feats), 0.0)
    pmask = np.zeros((T, P_MAX), dtype=bool)
    pmask[:, :P] = gmask

    close = (np.nan_to_num(dist, nan=1e9) < 5.0) & valid
    n_home = (close & (team[None] == 0)).sum(axis=1)
    n_away = (close & (team[None] == 1)).sum(axis=1)
    bxy = np.nan_to_num(ball_xy)
    ball = np.stack([
        bxy[:, 0] / HL, bxy[:, 1] / HW, bv[:, 0] / 20, bv[:, 1] / 20, bvis.astype(np.float64),
        np.minimum(since, 250) / 50.0, ref[:, 0] / HL, ref[:, 1] / HW, n_home / 3.0, n_away / 3.0,
    ], axis=-1).astype(np.float32)
    return players, pmask, ball
