"""Move decoded possession boundaries onto ball touches.

A possession changes hands when a player of the new team touches the ball, not a moment before or
after, and a restart is the touch that puts the ball back in play. The Viterbi decoder places a
boundary where the evidence flips, which can be a fraction of a second off. This post-step moves
each team switch (H<->A) and each restart (D -> H/A) to the first touch by a player of the NEW
team within +-W frames, if both neighbouring segments keep their minimum duration.

Touch = the ball changes velocity sharply (|v(t+3) - v(t-3)| >= `dv` m/s) next to a player (within
`radius` m). The values follow the public DSBA write-up (more-than-a-hack 2027); they are not tuned
here. Study and decision rule: WORKLOG.md, 2026-11-15.
"""

import numpy as np

from .common import A, D, H


def touches(pxy, team, bxy, fps=25, dv=4.0, radius=2.0, k=3):
    """(frames, side): touch frames and the team (0/1) of the nearest player at that frame."""
    v = np.diff(bxy, axis=0, prepend=bxy[:1]) * fps               # (T, 2), NaN where unseen
    jump = np.full(len(bxy), np.nan)
    jump[k:-k] = np.linalg.norm(v[2 * k:] - v[:-2 * k], axis=1)   # |v(t+k) - v(t-k)|
    d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)            # (T, S)
    d = np.where(np.isnan(d), np.inf, d)
    near = d.argmin(axis=1)
    ok = (jump >= dv) & (d[np.arange(len(d)), near] <= radius)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return idx, np.empty(0, dtype=np.int8)
    # one touch per burst of consecutive frames: the frame with the largest jump
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    frames = np.array([g[np.argmax(jump[g])] for g in groups])
    return frames, team[near[frames]].astype(np.int8)


def snap(codes, runs, frames, side, min_dur, W):
    """Copy of `codes` with team switches and restarts snapped to touches within +-W frames."""
    out = codes.copy()
    if W <= 0 or not len(frames):
        return out
    for a, b in runs:
        w = out[a:b]
        cut = np.flatnonzero(w[1:] != w[:-1]) + 1
        starts = np.concatenate(([0], cut)).astype(np.int64)
        ends = np.concatenate((cut, [len(w)])).astype(np.int64)
        for i in range(1, len(starts)):
            p, q = w[starts[i - 1]], w[starts[i]]
            if not ((p in (H, A) and q in (H, A)) or (p == D and q in (H, A))):
                continue
            t0 = a + starts[i]
            hit = frames[(frames >= t0 - W) & (frames <= t0 + W) & (side == q)]
            if not len(hit):
                continue
            new = int(hit[0]) - a                       # earliest touch by the new team
            if (new - starts[i - 1] >= min(min_dur[p], starts[i] - starts[i - 1])
                    and ends[i] - new >= min(min_dur[q], ends[i] - starts[i])):
                w[min(new, starts[i]):max(new, starts[i])] = q if new < starts[i] else p
                starts[i] = new
    return out
