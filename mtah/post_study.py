"""Study of two decoder add-ons on IDSSE validation, with the adoption rule applied in code.

    python -m mtah.post_study                       # ensemble run2+run3, the 8 stress scenarios

Rule (pre-registered in WORKLOG.md, 2026-11-15): an option passes if, against the baseline, the mean
spellF1 over the scenarios is up by >= 0.005 AND atd_like is up AND base is not down by more than
0.01. Best mean among passing options; ties -> the smaller |b| / W. Study 2 (touch snapping) runs on
top of Study 1's outcome (D bias b = 0 unless one passed). Nothing else is tried.
"""

import argparse
import json

import numpy as np

from .gbdt import load_matches
from .make_submission import load_decoder
from .decode import probs_to_codes
from .scoring import score
from .snap import snap, touches
from .stress import SCENARIOS, scenario_inputs, spec_probs

D_BIAS = (-0.3, -0.15, 0.15, 0.3)
WINDOWS = (12, 25, 50)


def passes(rows, base):
    mean, ref = np.mean(list(rows.values())), np.mean(list(base.values()))
    return mean >= ref + 0.005 and rows["atd_like"] > base["atd_like"] and rows["base"] >= base["base"] - 0.01


def pick(options, base):
    """options: {key: {scenario: spellF1}}; key = |b| or W for tie-breaking, returns key or None."""
    ok = [k for k, r in options.items() if passes(r, base)]
    if not ok:
        return None
    return min(ok, key=lambda k: (-round(np.mean(list(options[k].values())), 6), abs(k)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="outputs/nn/run2+outputs/nn/run3")
    ap.add_argument("--val", nargs="+", default=["idsse_J03WMX", "idsse_J03WR9"])
    ap.add_argument("--out", default="outputs/post_study.json")
    args = ap.parse_args()

    ms = [m for m in load_matches("cache/idsse_*.npz", 25) if m.name in args.val]
    dec = load_decoder(args.model.split("+")[0] + "/decoder.json")
    mind = dec["min_dur"]
    gt = [m.gt_codes() for m in ms]
    data = {}
    for name, spec in SCENARIOS.items():
        for k, m in enumerate(ms):
            inp = scenario_inputs(m, k, spec)
            data[name, k] = (spec_probs(args.model, name, m, inp), inp)

    def run(b=0.0, W=0):
        res, shares = {}, {}
        for name in SCENARIOS:
            sp, dr = [], []
            for k, m in enumerate(ms):
                probs, (pxy, team, bxy) = data[name, k]
                codes = probs_to_codes(probs, m.runs(), d_bias=b, **dec)
                if W:
                    f, s = touches(pxy, team, bxy, m.fps)
                    codes = snap(codes, m.runs(), f, s, mind, W)
                sp.append(score(gt[k], codes)["spell_f1_mean"])
                g = gt[k] != 3
                lab = codes[g]
                dr.append([np.mean(lab == c) for c in (0, 1, 2)] + [np.mean(gt[k][g] == c) for c in (0, 1, 2)])
            res[name] = float(np.mean(sp))
            shares[name] = np.mean(dr, axis=0).round(3).tolist()
        return res, shares

    base, base_shares = run()
    print("baseline spellF1 per scenario:", {k: round(v, 4) for k, v in base.items()},
          f"mean {np.mean(list(base.values())):.4f}")
    print("\nStudy 1a - graded-frame shares, predicted H/A/D | ground truth H/A/D")
    for name, s in base_shares.items():
        print(f"  {name:<14} pred {s[0]:.3f} {s[1]:.3f} {s[2]:.3f} | gt {s[3]:.3f} {s[4]:.3f} {s[5]:.3f}")

    print("\nStudy 1b - D bias")
    d_opts = {}
    for b in D_BIAS:
        d_opts[b], _ = run(b=b)
        print(f"  b {b:+.2f}  mean {np.mean(list(d_opts[b].values())):.4f}  atd_like {d_opts[b]['atd_like']:.4f}"
              f"  base {d_opts[b]['base']:.4f}  pass {passes(d_opts[b], base)}")
    b_star = pick(d_opts, base)
    b_star = 0.0 if b_star is None else b_star
    print(f"  -> adopted b = {b_star}")

    ref, _ = run(b=b_star)
    print(f"\nStudy 2 - touch snapping on top of b = {b_star} (reference mean {np.mean(list(ref.values())):.4f})")
    w_opts = {}
    for W in WINDOWS:
        w_opts[W], _ = run(b=b_star, W=W)
        print(f"  W {W:3d}  mean {np.mean(list(w_opts[W].values())):.4f}  atd_like {w_opts[W]['atd_like']:.4f}"
              f"  base {w_opts[W]['base']:.4f}  pass {passes(w_opts[W], ref)}")
    w_star = pick(w_opts, ref) or 0
    print(f"  -> adopted W = {w_star}")

    with open(args.out, "w") as fh:
        json.dump(dict(baseline=base, shares=base_shares, d_bias={str(k): v for k, v in d_opts.items()},
                       b_star=b_star, snap={str(k): v for k, v in w_opts.items()}, W_star=w_star), fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
