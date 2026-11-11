"""Compare the noise of the real ATD with our simulated corruption of IDSSE -- inputs only, no labels.

Measuring the input is allowed (it uses no ground truth). Every number printed is an aggregate
(fractions, means, quantiles): nothing frame-level is shown, so the output is safe to share.

    python -m mtah.noise_stats
"""

import numpy as np

from .augment import CorruptCfg, corrupt
from .common import Match

HL, HW = 52.5, 34.0


def _runs_of(mask):
    """Lengths of maximal True runs."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[1::2] - edges[::2]


def stats(players_xy, team, ball_xy, period, fps=25):
    g = period != 0
    pxy, bxy = players_xy[g], ball_xy[g]
    seen = ~np.isnan(bxy[:, 0])
    gaps = _runs_of(~seen)
    vis = _runs_of(seen)
    step = np.linalg.norm(np.diff(bxy, axis=0), axis=1) * fps          # m/s between frames
    step = step[~np.isnan(step)]
    valid = ~np.isnan(pxy[..., 0])
    n_obj = valid.sum(1)
    n_home = (valid & (team[None] == 0)).sum(1)
    n_away = (valid & (team[None] == 1)).sum(1)
    track_len = np.concatenate([_runs_of(valid[:, s]) for s in range(valid.shape[1])])
    out_ball = (np.abs(bxy[seen, 0]) > HL + 1) | (np.abs(bxy[seen, 1]) > HW + 1)
    d = np.linalg.norm(pxy - np.nan_to_num(bxy)[:, None], axis=2)
    nearest = np.nanmin(np.where(valid, d, np.nan), axis=1)[seen]
    return {
        "ball seen": f"{seen.mean():.1%}",
        "ball gap len (frames) median/mean/p90": f"{np.median(gaps):.0f} / {gaps.mean():.1f} / "
                                                 f"{np.percentile(gaps, 90):.0f}" if len(gaps) else "-",
        "ball visible run median/mean": f"{np.median(vis):.0f} / {vis.mean():.1f}" if len(vis) else "-",
        "ball jumps > 40 m/s (per seen frame)": f"{np.mean(step > 40):.2%}",
        "ball jumps > 15 m between frames": f"{np.mean(step > 15 * fps):.3%}",
        "ball seen outside pitch (+1 m)": f"{out_ball.mean():.1%}",
        "nearest player to ball median / p90 (m)": f"{np.nanmedian(nearest):.1f} / "
                                                   f"{np.nanpercentile(nearest, 90):.1f}",
        "slots": str(valid.shape[1]),
        "objects/frame mean (home, away)": f"{n_obj.mean():.1f} ({n_home.mean():.1f}, {n_away.mean():.1f})",
        "frames with > 22 objects": f"{np.mean(n_obj > 22):.1%}",
        "frames with < 18 objects": f"{np.mean(n_obj < 18):.1%}",
        "track segment len (s) median / mean": f"{np.median(track_len) / fps:.1f} / {track_len.mean() / fps:.1f}",
    }


def main():
    from . import atd
    m = atd.load()
    cols = {"ATD (BAR-ATH)": stats(m.players_xy, m.team, m.ball_xy, m.period, m.fps)}
    ref = Match.load("cache/idsse_J03WR9.npz")
    for name, cfg in [("IDSSE base", CorruptCfg(flip=False, team_swap=False)),
                      ("IDSSE +frag+false", CorruptCfg(flip=False, team_swap=False,
                                                        fragment=True, false_ball=0.03))]:
        z = ref.ball_extra[:, :1] if ref.ball_extra is not None else None
        p, t, b, _ = corrupt(ref.players_xy, ref.team, ref.ball_xy, None,
                             np.random.default_rng(0), cfg, z)
        cols[name] = stats(p, t, b, ref.period, ref.fps)
    keys = list(next(iter(cols.values())))
    w = max(map(len, keys))
    print(f"{'':<{w}}  " + "  ".join(f"{c:>22}" for c in cols))
    for k in keys:
        print(f"{k:<{w}}  " + "  ".join(f"{cols[c][k]:>22}" for c in cols))


if __name__ == "__main__":
    main()
