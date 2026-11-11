"""The official baseline algorithm, re-run on our IDSSE validation set.

Scores on IDSSE are not comparable to the board: different matches, our own corruption, our own
label cleaning. What IS comparable is the gap between a method and this baseline on the same
validation data. The baseline scores spellF1 0.2433 on the real match; a method that beats it by a
wide margin here has a fair chance of doing so there.

    python -m mtah.reference
"""

import numpy as np

from .augment import CorruptCfg, corrupt
from .common import D
from .gbdt import ball_z, load_matches
from .scoring import brief, score


def nearest_player_codes(players_xy, team, ball_xy):
    """starter/baseline.py step 1: team of the nearest player where the ball is seen, else none."""
    d = np.linalg.norm(players_xy - ball_xy[:, None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    out = team[np.argmin(d, axis=1)].astype(np.int8)
    out[np.isnan(ball_xy[:, 0]) | np.isinf(d.min(1))] = -1
    return out


def smooth(codes, window):
    """starter/baseline.py step 2: centred majority vote, D where nobody was seen."""
    if window <= 0:
        return np.where(codes < 0, D, codes).astype(np.int8)
    ch = np.concatenate(([0], np.cumsum(codes == 0)))
    ca = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = ch[hi] - ch[lo], ca[hi] - ca[lo]
    return np.where(h + a == 0, D, np.where(h >= a, 0, 1)).astype(np.int8)


def main(val=("idsse_J03WMX", "idsse_J03WR9"), seed=1234):
    cfg = CorruptCfg(flip=False, team_swap=False)
    ms = [m for m in load_matches("cache/idsse_*.npz", 25) if m.name in val]
    for w in (0, 25, 50, 125):
        res = []
        for k, m in enumerate(ms):
            pxy, team, bxy, _ = corrupt(m.players_xy, m.team, m.ball_xy, None,
                                        np.random.default_rng([seed, k]), cfg, ball_z(m))
            codes = smooth(nearest_player_codes(pxy, team, bxy), w)
            codes[m.period == 0] = 3
            r = score(m.gt_codes(), codes)
            res.append(r["spell_f1_mean"])
            print(f"  smooth{w:<4} {m.name:<14} {brief(r)}")
        print(f"baseline smooth{w}: mean spellF1 {np.mean(res):.4f}")


if __name__ == "__main__":
    main()
