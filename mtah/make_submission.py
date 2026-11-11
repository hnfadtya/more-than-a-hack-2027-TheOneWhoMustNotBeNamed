"""BAR-ATH end to end: ATD -> model probabilities -> decoder -> validated submission CSV.

Runs locally on CPU; the competition data never leaves the machine.

    python -m mtah.make_submission --model outputs/nn/run1 --name nn_run1
    python -m mtah.make_submission --model outputs/gbdt/v1 --name gbdt_v1
    python -m mtah.make_submission --model outputs/nn/run1 outputs/gbdt/v1 --weights 0.75 0.25 \
        --decoder outputs/nn/run1/decoder.json --name ens

A model directory holds either best.pt (network) or model.pkl (GBDT), plus decoder.json (decoder
settings tuned for that model on IDSSE validation). Probabilities are cached next to the model so
re-decoding is instant.

The public 5-minute window is printed as a PIPELINE CHECK only (alignment, labels mean what we
think). It is never used to choose a model or a parameter -- RULES.md.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from .common import ROOT, X, write_submission
from .decode import probs_to_codes
from .scoring import brief, pm, public_gt, score


def model_probs(model_dir, m, refresh=False):
    cache = os.path.join(model_dir, f"{m.name}_probs.npz")
    if os.path.isfile(cache) and not refresh:
        return np.load(cache)["probs"]
    t0 = time.time()
    if os.path.isfile(os.path.join(model_dir, "best.pt")):
        import torch
        from .infer import load_model, predict_match
        torch.manual_seed(0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        net, _ = load_model(os.path.join(model_dir, "best.pt"))
        probs = predict_match(net, m.players_xy, m.team, m.ball_xy, m.runs(), m.fps)
    elif os.path.isfile(os.path.join(model_dir, "model.pkl")):
        import pickle
        from .gbdt import predict_probs
        from .handfeat import match_features
        with open(os.path.join(model_dir, "model.pkl"), "rb") as fh:
            gbdt = pickle.load(fh)
        Xf, rows = match_features(m.players_xy, m.team, m.ball_xy, m.runs(), m.fps)
        probs = predict_probs(gbdt, Xf, rows, m.n_frames)
    else:
        raise SystemExit(f"{model_dir}: no best.pt or model.pkl")
    np.savez_compressed(cache, probs=probs)
    print(f"  {model_dir}: probabilities in {time.time() - t0:.0f}s -> {cache}")
    return probs


def load_decoder(path):
    with open(path) as fh:
        c = json.load(fh)
    sw = np.full((3, 3), c["sw_dead"])
    sw[0, 1] = sw[1, 0] = c["sw_team"]
    return dict(min_dur=tuple(c["min_dur"]), switch=sw)


def check(path, n_frames):
    """Exactly the official checks (evaluation/validate_submission.py)."""
    reason = pm.validate_submission(pm.read_label_csv(path), n_frames)
    if reason:
        raise SystemExit(f"INVALID submission {path}: {reason}")
    print(f"  valid: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", required=True, help="model dir(s)")
    ap.add_argument("--weights", nargs="+", type=float, help="ensemble weights (default equal)")
    ap.add_argument("--decoder", help="decoder.json (default: the first model's)")
    ap.add_argument("--name", required=True, help="file name: evaluation/results/<name>.csv")
    ap.add_argument("--refresh", action="store_true", help="recompute cached probabilities")
    args = ap.parse_args()

    from . import atd
    m = atd.load()
    w = np.array(args.weights or [1.0] * len(args.model), dtype=np.float64)
    w = w / w.sum()
    probs = sum(wi * model_probs(d, m, args.refresh).astype(np.float64)
                for wi, d in zip(w, args.model))

    dec = load_decoder(args.decoder or os.path.join(args.model[0], "decoder.json"))
    codes = probs_to_codes(probs, m.runs(), **dec)
    codes[m.period == 0] = X

    out = os.path.join(ROOT, "evaluation", "results", f"{args.name}.csv")
    write_submission(out, codes)
    check(out, m.n_frames)

    graded = m.period != 0
    gt_like = np.where(graded, 0, X).astype(np.int8)   # only to get the graded runs
    st = pm.possession_stats(codes, pm.valid_runs(gt_like))
    lab = codes[graded]
    print(f"  graded frames: H {np.mean(lab == 0):.1%}  A {np.mean(lab == 1):.1%}  "
          f"D {np.mean(lab == 2):.1%}  | possessions {st['n_spells']}, "
          f"mean {st['mean_duration_s']:.1f}s, home share {st['home_share']:.2f}")
    print("  pipeline check on the public 5 min (NOT a score to optimise):",
          brief(score(public_gt(), codes)))


if __name__ == "__main__":
    sys.exit(main())
