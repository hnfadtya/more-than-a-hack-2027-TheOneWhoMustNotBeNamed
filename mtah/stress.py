"""Robustness test: how much does each model lose when the tracking is worse than in training?

We cannot measure anything on BAR-ATH, but we can measure which model degrades more gracefully
when the IDSSE validation matches are corrupted in ways the models never saw: more missing ball,
false ball detections, fragmented tracks with slot reuse, tracklets on the wrong team, duplicates.
The ATD has all of these to some degree; a model whose score collapses under them is a risk.

    python -m mtah.stress --models outputs/nn/run1 outputs/gbdt/v1
"""

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np

from .augment import CorruptCfg, add_false_ball, corrupt, fragment_tracks
from .gbdt import ball_z, load_matches
from .make_submission import load_decoder
from .decode import probs_to_codes
from .scoring import score

BASE = CorruptCfg(flip=False, team_swap=False)
SCENARIOS = {
    "base": dict(),
    "ball_heavy": dict(cfg=dict(ball_missing=0.55, ball_burst=80.0)),
    "players_heavy": dict(cfg=dict(player_missing=0.15, player_burst=100.0, player_jitter=0.5)),
    "false_ball": dict(false_ball=0.04),
    "fragment": dict(fragment=True),
    "harsh": dict(cfg=dict(ball_missing=0.55, ball_burst=80.0, player_missing=0.15,
                           player_burst=100.0, player_jitter=0.5), false_ball=0.04, fragment=True),
}


def scenario_inputs(m, k, spec, seed=4321):
    rng = np.random.default_rng([seed, k])
    cfg = replace(BASE, **spec.get("cfg", {}))
    pxy, team, bxy, _ = corrupt(m.players_xy, m.team, m.ball_xy, None, rng, cfg, ball_z(m))
    if spec.get("false_ball"):
        bxy = add_false_ball(bxy, pxy, rng, spec["false_ball"])
    if spec.get("fragment"):
        pxy, team = fragment_tracks(pxy, team, rng)
    return pxy, team, bxy


def model_probs(model_dir, m, pxy, team, bxy):
    if os.path.isfile(os.path.join(model_dir, "best.pt")):
        import torch
        from .infer import load_model, predict_match
        torch.set_num_threads(os.cpu_count() or 4)
        net, _ = load_model(os.path.join(model_dir, "best.pt"))
        return predict_match(net, pxy, team, bxy, m.runs(), m.fps)
    import pickle
    from .gbdt import predict_probs
    from .handfeat import match_features
    with open(os.path.join(model_dir, "model.pkl"), "rb") as fh:
        gbdt = pickle.load(fh)
    X, rows = match_features(pxy, team, bxy, m.runs(), m.fps)
    return predict_probs(gbdt, X, rows, m.n_frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=["idsse_J03WMX", "idsse_J03WR9"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", default="outputs/stress.json")
    args = ap.parse_args()

    ms = [m for m in load_matches("cache/idsse_*.npz", 25) if m.name in args.val]
    decoders = {d: load_decoder(os.path.join(d, "decoder.json")) for d in args.models}
    table = {}
    for name in args.scenarios:
        spec = SCENARIOS[name]
        table[name] = {}
        inputs = [scenario_inputs(m, k, spec) for k, m in enumerate(ms)]
        n_slots = int(np.mean([inp[0].shape[1] for inp in inputs]))
        ball_seen = float(np.mean([np.mean(~np.isnan(inp[2][:, 0])) for inp in inputs]))
        for d in args.models:
            t0 = time.time()
            spells, poss = [], []
            for m, (pxy, team, bxy) in zip(ms, inputs):
                codes = probs_to_codes(model_probs(d, m, pxy, team, bxy), m.runs(), **decoders[d])
                r = score(m.gt_codes(), codes)
                spells.append(r["spell_f1_mean"])
                poss.append(f"{r['n_pred_spells']}/{r['n_gt_spells']}")
            table[name][d] = float(np.mean(spells))
            print(f"{name:<14} {d:<20} spellF1 {np.mean(spells):.4f}  poss {' '.join(poss)}  "
                  f"(slots {n_slots}, ball seen {ball_seen:.0%}, {time.time() - t0:.0f}s)", flush=True)
    print("\nscenario       " + "  ".join(f"{d:>20}" for d in args.models))
    for name, row in table.items():
        print(f"{name:<14} " + "  ".join(f"{row[d]:>20.4f}" for d in args.models))
    with open(args.out, "w") as fh:
        json.dump(table, fh, indent=1)


if __name__ == "__main__":
    main()
