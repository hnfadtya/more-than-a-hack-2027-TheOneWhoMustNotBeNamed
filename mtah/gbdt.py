"""Jalur A: hand-designed features (mtah.handfeat) -> gradient boosting -> Viterbi decoder.

The features are football rules; the boosting only learns how to weigh them, from OTHER games
(IDSSE), corrupted to look like the ATD. CPU only, a few minutes, deterministic.

    # train, validate on held-out IDSSE matches, save model + validation probabilities
    python -m mtah.gbdt train --val idsse_J03WMX idsse_J03WR9 --out outputs/gbdt/v1

    # per-frame probabilities for BAR-ATH (competition data, local only)
    python -m mtah.gbdt predict --model outputs/gbdt/v1/model.pkl
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .augment import CorruptCfg, corrupt
from .common import Match
from .decode import DEFAULT_DECODE, probs_to_codes
from .handfeat import match_features
from .labels import absorb_short_spells
from .scoring import brief, score


def ball_z(m):
    return m.ball_extra[:, :1] if m.ball_extra is not None and m.ball_extra.shape[1] else None


def load_matches(pattern, min_spell):
    ms = [Match.load(p) for p in sorted(glob.glob(pattern))]
    for m in ms:
        m.label = absorb_short_spells(m.label, m.runs(), min_spell)
    return ms


def corrupted_features(m, cfg, rng):
    pxy, team, bxy, lab = corrupt(m.players_xy, m.team, m.ball_xy, m.label, rng, cfg, ball_z(m))
    X, rows = match_features(pxy, team, bxy, m.runs(), m.fps)
    return X, rows, lab


def build_train(matches, cfg, copies, stride, seed):
    Xs, ys = [], []
    for i, m in enumerate(matches):
        for c in range(copies):
            rng = np.random.default_rng([seed, i, c])
            X, rows, lab = corrupted_features(m, cfg, rng)
            y = lab[rows]
            keep = (y < 3) & ((rows + c) % stride == 0)
            Xs.append(X[keep])
            ys.append(y[keep])
            print(f"  {m.name} copy {c}: {int(keep.sum())} rows")
    import pandas as pd
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def predict_probs(model, X, rows, n_frames):
    probs = np.full((n_frames, 3), np.nan, dtype=np.float32)
    probs[rows] = model.predict_proba(X[model.feature_names_in_])
    return probs


def val_set(matches, cfg, seed=1234):
    """Each validation match corrupted once with a fixed seed (no flips/swaps): comparable runs."""
    vcfg = CorruptCfg(**{**cfg.to_dict(), "flip": False, "team_swap": False})
    out = []
    for k, m in enumerate(matches):
        X, rows, _ = corrupted_features(m, vcfg, np.random.default_rng([seed, k]))
        out.append((m, X, rows))
    return out


def train(args):
    cfg = CorruptCfg()
    ms = load_matches(args.data, args.min_spell)
    tr = [m for m in ms if m.name not in args.val]
    va = [m for m in ms if m.name in args.val]
    print(f"train {[m.name for m in tr]}\nval   {[m.name for m in va]}")

    t0 = time.time()
    X, y = build_train(tr, cfg, args.copies, args.stride, args.seed)
    print(f"train rows {len(y)}, features {X.shape[1]}, labels H/A/D "
          f"{np.bincount(y, minlength=3).tolist()}  ({time.time() - t0:.0f}s)")

    model = HistGradientBoostingClassifier(
        max_iter=args.iters, learning_rate=args.lr, max_leaf_nodes=args.leaves,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
    t0 = time.time()
    model.fit(X, y)
    print(f"fit {time.time() - t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    results, probs = {}, {}
    for m, Xv, rows in val_set(va, cfg):
        p = predict_probs(model, Xv, rows, m.n_frames)
        codes = probs_to_codes(p, m.runs(), **DEFAULT_DECODE)
        gt = m.gt_codes()
        r = score(gt, codes)
        g = gt < 3
        r["frame_acc_raw"] = float((np.argmax(np.nan_to_num(p), 1)[g] == gt[g]).mean())
        results[m.name], probs[m.name] = r, p
        print(f"  {m.name:<14} {brief(r)}  rawAcc {r['frame_acc_raw']:.3f}")
    spell = float(np.mean([r["spell_f1_mean"] for r in results.values()]))
    print(f"val spellF1 (default decoder) {spell:.4f}")

    with open(os.path.join(args.out, "model.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    np.savez_compressed(os.path.join(args.out, "val_probs.npz"), **probs)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"args": vars(args), "corrupt": cfg.to_dict(), "val_spellF1": spell,
                   "val": {k: {kk: v[kk] for kk in ("spell_f1_mean", "macro_f1", "n_pred_spells",
                                                    "n_gt_spells", "frame_acc_raw")}
                           for k, v in results.items()}}, fh, indent=1)

    imp = permutation_free_importance(model, X.columns)
    if imp:
        print("top features:", ", ".join(imp[:12]))


def permutation_free_importance(model, columns):
    """Rough feature ranking without extra passes: how often a feature is split on."""
    counts = {}
    try:
        for stage in model._predictors:
            for tree in stage:
                for node in tree.nodes:
                    if not node["is_leaf"]:
                        c = columns[node["feature_idx"]]
                        counts[c] = counts.get(c, 0) + 1
    except Exception:
        return []
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def predict(args):
    from . import atd
    with open(args.model, "rb") as fh:
        model = pickle.load(fh)
    m = atd.load()
    X, rows = match_features(m.players_xy, m.team, m.ball_xy, m.runs(), m.fps)
    probs = predict_probs(model, X, rows, m.n_frames)
    out = args.out or os.path.join(os.path.dirname(args.model), f"{m.name}_probs.npz")
    np.savez_compressed(out, probs=probs)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="cache/idsse_*.npz")
    t.add_argument("--val", nargs="+", default=["idsse_J03WMX", "idsse_J03WR9"])
    t.add_argument("--out", required=True)
    t.add_argument("--min_spell", type=int, default=25)
    t.add_argument("--copies", type=int, default=2, help="corrupted copies per training match")
    t.add_argument("--stride", type=int, default=3, help="keep every n-th frame")
    t.add_argument("--iters", type=int, default=300)
    t.add_argument("--lr", type=float, default=0.1)
    t.add_argument("--leaves", type=int, default=63)
    t.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else predict(args)


if __name__ == "__main__":
    main()
