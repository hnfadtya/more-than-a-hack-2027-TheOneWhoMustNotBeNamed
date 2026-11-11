"""Grid-search the Viterbi decoder on IDSSE validation probabilities (never on BAR-ATH).

Works for any model that saved `val_probs.npz` ({match name: (T,3) probs}): GBDT or the network.
The switch cost is split in two: between teams (a turnover) and to/from dead ball (a stoppage or
a restart), because those are different events with different evidence.

    python -m mtah.tune_decoder outputs/gbdt/v1/val_probs.npz
"""

import argparse
import itertools
import json
import os

import numpy as np

from .decode import probs_to_codes
from .gbdt import load_matches
from .scoring import score

MIN_TEAM = (1, 12, 25, 50)
MIN_DEAD = (12, 25, 50, 100)
SW_TEAM = (0.0, 1.0, 2.0, 4.0, 8.0)
SW_DEAD = (0.0, 1.0, 2.0, 4.0, 8.0)


def switch_matrix(team, dead):
    s = np.full((3, 3), dead)
    s[0, 1] = s[1, 0] = team
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("probs")
    ap.add_argument("--min_spell", type=int, default=25)
    ap.add_argument("--min_team", type=int, nargs="+", default=MIN_TEAM)
    ap.add_argument("--min_dead", type=int, nargs="+", default=MIN_DEAD)
    ap.add_argument("--sw_team", type=float, nargs="+", default=SW_TEAM)
    ap.add_argument("--sw_dead", type=float, nargs="+", default=SW_DEAD)
    ap.add_argument("--tag", default="", help="suffix for the output json")
    args = ap.parse_args()

    probs = dict(np.load(args.probs))
    ms = [m for m in load_matches("cache/idsse_*.npz", args.min_spell) if m.name in probs]
    rows = []
    for mt, md, st, sd in itertools.product(args.min_team, args.min_dead, args.sw_team, args.sw_dead):
        cfg = dict(min_dur=(mt, mt, md), switch=switch_matrix(st, sd))
        s = [score(m.gt_codes(), probs_to_codes(probs[m.name], m.runs(), **cfg)) for m in ms]
        rows.append({"min_team": mt, "min_dead": md, "sw_team": st, "sw_dead": sd,
                     "spellF1": float(np.mean([r["spell_f1_mean"] for r in s])),
                     "macroF1": float(np.mean([r["macro_f1"] for r in s])),
                     "per_match": [round(r["spell_f1_mean"], 4) for r in s]})
    rows.sort(key=lambda r: -r["spellF1"])
    print(f"{len(rows)} configs on {[m.name for m in ms]}")
    for r in rows[:10]:
        print(f"  spellF1 {r['spellF1']:.4f}  macroF1 {r['macroF1']:.4f}  min_team {r['min_team']:3d}"
              f"  min_dead {r['min_dead']:3d}  sw_team {r['sw_team']:.1f}  sw_dead {r['sw_dead']:.1f}"
              f"  {r['per_match']}")
    out = os.path.join(os.path.dirname(args.probs), f"decoder_grid{args.tag}.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
