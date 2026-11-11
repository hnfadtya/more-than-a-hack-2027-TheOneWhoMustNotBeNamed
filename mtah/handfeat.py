"""Hand-designed per-frame features for possession and dead ball (Jalur A).

Every feature is a football rule of thumb that works on identity-free, gappy tracking:

  ball            seen?, frames since last / until next sighting, length of the current gap,
                  position estimate (linear across gaps), speed, distance outside the pitch,
                  distance to touchline / goal line
  who is close    nearest home / away player to the ball, their difference (the control margin),
                  how many of each within 2 m and 5 m, the team nearest at the last and at the
                  next sighting (what a gap most likely hides)
  how they move   mean speed of home / away / everyone (players walk when the ball is dead),
                  number of tracked players, distance between team centroids
  context         centred rolling means / mins / maxes of the key signals over 0.5 s, 2 s, 5 s

Invariant to flipping the pitch (only |x|, |y| and distances are used), and H/A-symmetric under the
team swap used in training: every home feature has its away twin.
"""

import numpy as np
import pandas as pd

HL, HW = 52.5, 34.0
FAR = 99.0
BIG = 10 ** 6
WINDOWS = (12, 50, 125)          # frames: ~0.5 s, 2 s, 5 s at 25 fps


def _ffill_idx(seen):
    T = len(seen)
    return np.maximum.accumulate(np.where(seen, np.arange(T), -1))


def _bfill_idx(seen):
    T = len(seen)
    idx = np.where(seen, np.arange(T), BIG)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _ffill(values, seen):
    i = _ffill_idx(seen)
    return np.where(i >= 0, values[np.maximum(i, 0)], np.nan)


def _bfill(values, seen):
    i = _bfill_idx(seen)
    T = len(seen)
    return np.where(i < BIG, values[np.minimum(i, T - 1)], np.nan)


def _speed(xy, fps):
    """Central-difference speed (T, ...) in m/s; NaN where a neighbour is missing."""
    v = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) >= 3:
        v[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=-1) * (fps / 2.0)
    return v


def _nanmean(a, axis):
    with np.errstate(invalid="ignore"):
        n = (~np.isnan(a)).sum(axis)
        return np.where(n > 0, np.nansum(a, axis) / np.maximum(n, 1), np.nan)


def frame_features(players_xy, team, ball_xy, fps=25):
    """Per-frame feature DataFrame for one contiguous block (one half)."""
    T = len(ball_xy)
    t = np.arange(T)
    valid = ~np.isnan(players_xy[..., 0])
    home = (team == 0)[None, :] & valid
    away = (team == 1)[None, :] & valid

    # -- ball: sightings, gaps, estimated position ---------------------------------------------
    seen = ~np.isnan(ball_xy[:, 0])
    ip, inx = _ffill_idx(seen), _bfill_idx(seen)
    since = np.where(ip >= 0, t - ip, BIG).astype(np.float32)
    until = np.where(inx < BIG, inx - t, BIG).astype(np.float32)
    prev_xy = np.where((ip >= 0)[:, None], ball_xy[np.maximum(ip, 0)], np.nan)
    next_xy = np.where((inx < BIG)[:, None], ball_xy[np.minimum(inx, T - 1)], np.nan)
    w = np.where((ip >= 0) & (inx < BIG), since / np.maximum(since + until, 1), 0.0)[:, None]
    est = np.where(np.isnan(prev_xy), next_xy,
                   np.where(np.isnan(next_xy), prev_xy, (1 - w) * prev_xy + w * next_xy))
    est = np.where(seen[:, None], ball_xy, est)
    est0 = np.nan_to_num(est)

    bspeed = _speed(ball_xy, fps)
    bx, by = np.abs(est0[:, 0]), np.abs(est0[:, 1])

    # -- who is close ---------------------------------------------------------------------------
    d = np.linalg.norm(players_xy - est0[:, None], axis=2)
    dh = np.where(home, d, np.inf).min(1)
    da = np.where(away, d, np.inf).min(1)
    dh = np.minimum(dh, FAR)
    da = np.minimum(da, FAR)
    margin = da - dh                                       # > 0: home closer
    near_team = np.where(seen, np.sign(margin), np.nan)    # at sightings only

    # -- movement -------------------------------------------------------------------------------
    pspeed = np.clip(_speed(players_xy, fps), 0, 12)
    pspeed = np.where(valid, pspeed, np.nan)
    sp_home = _nanmean(np.where(home, pspeed, np.nan), 1)
    sp_away = _nanmean(np.where(away, pspeed, np.nan), 1)
    sp_all = _nanmean(pspeed, 1)
    cx_home = _nanmean(np.where(home, players_xy[..., 0], np.nan), 1)
    cx_away = _nanmean(np.where(away, players_xy[..., 0], np.nan), 1)

    f = {
        "seen": seen.astype(np.float32),
        "since": np.minimum(since, 500), "until": np.minimum(until, 500),
        "gap": np.minimum(np.where(seen, 0, since + until), 1000),
        "ball_speed": bspeed,
        "ball_ax": bx, "ball_ay": by,
        "out_x": np.maximum(bx - HL, 0), "out_y": np.maximum(by - HW, 0),
        "to_goalline": HL - bx, "to_touchline": HW - by,
        "dh": dh, "da": da, "dmin": np.minimum(dh, da), "margin": np.clip(margin, -30, 30),
        "nh2": (home & (d < 2)).sum(1), "na2": (away & (d < 2)).sum(1),
        "nh5": (home & (d < 5)).sum(1), "na5": (away & (d < 5)).sum(1),
        "near_prev": _ffill(near_team, seen), "near_next": _bfill(near_team, seen),
        "margin_prev": np.clip(_ffill(margin, seen), -30, 30),
        "margin_next": np.clip(_bfill(margin, seen), -30, 30),
        "sp_home": sp_home, "sp_away": sp_away, "sp_all": sp_all,
        "n_tracked": valid.sum(1), "n_home": home.sum(1), "n_away": away.sum(1),
        "centroid_gap": np.abs(cx_home - cx_away),
    }
    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})

    # -- temporal context -------------------------------------------------------------------------
    roll_mean = ["seen", "margin", "dmin", "sp_all", "sp_home", "sp_away", "ball_speed",
                 "out_x", "out_y", "nh5", "na5"]
    roll_ext = ["ball_speed", "dmin", "sp_all"]
    ctx = {}
    for wlen in WINDOWS:
        r = df[roll_mean].rolling(2 * wlen + 1, center=True, min_periods=1)
        m = r.mean()
        ctx.update({f"{c}_m{wlen}": m[c].values for c in roll_mean})
        r2 = df[roll_ext].rolling(2 * wlen + 1, center=True, min_periods=1)
        mx, mn = r2.max(), r2.min()
        ctx.update({f"{c}_max{wlen}": mx[c].values for c in roll_ext})
        ctx.update({f"{c}_min{wlen}": mn[c].values for c in roll_ext})
        # sign of the control margin, averaged: who has been closer over the window
        s = pd.Series(np.sign(df["margin"].values)).rolling(2 * wlen + 1, center=True,
                                                            min_periods=1).mean()
        ctx[f"side_m{wlen}"] = s.values
    return pd.concat([df, pd.DataFrame(ctx, dtype=np.float32)], axis=1)


def match_features(players_xy, team, ball_xy, runs, fps=25):
    """Features for every frame inside `runs`, plus the row index into the match."""
    parts, rows = [], []
    for a, b in runs:
        parts.append(frame_features(players_xy[a:b], team, ball_xy[a:b], fps))
        rows.append(np.arange(a, b))
    return pd.concat(parts, ignore_index=True), np.concatenate(rows)
