"""Robustness test: how much does each model lose when the tracking is worse than in training?

We cannot measure anything on BAR-ATH, but we can measure which model degrades more gracefully
when the IDSSE validation matches are corrupted in ways the models never saw: more missing ball,
false ball detections, fragmented tracks with slot reuse, tracklets on the wrong team, duplicates.
The ATD has all of these to some degree; a model whose score collapses under them is a risk.

    python -m mtah.stress --models outputs/nn/run1 outputs/gbdt/v1
    python -m mtah.stress --models outputs/nn/run2 outputs/nn/run2+outputs/nn/run3   # ensemble

A model spec "dirA+dirB" averages the models' probabilities (equal weights) and decodes with dirA's
decoder.json; "dirA+dirB@path/decoder.json" decodes with that decoder instead. Per-model probabilities are cached in outputs/stress_cache/, keyed by the scenario
spec, so comparing ensembles does not recompute them.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace

import numpy as np

from .augment import CorruptCfg, add_false_ball, corrupt, fragment_tracks, vball_fill
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
    # matched to mtah.noise_stats on the real ATD: long ball gaps, ball lost when far from players,
    # ends of the pitch out of view
    "long_gaps": dict(cfg=dict(ball_burst=430.0)),
    "atd_like": dict(cfg=dict(ball_missing=0.22, ball_burst=430.0, ball_far_boost=2.0, ball_out_boost=8.0, ball_jitter=0.1,
                              fov_missing=0.08, fov_burst=150.0), fragment=True),
}


def scenario_inputs(m, k, spec, seed=4321):
    rng = np.random.default_rng([seed, k])
    cfg = replace(BASE, **spec.get("cfg", {}))
    pxy, team, bxy, _ = corrupt(m.players_xy, m.team, m.ball_xy, None, rng, cfg, ball_z(m))
    if spec.get("false_ball"):
        bxy = add_false_ball(bxy, pxy, rng, spec["false_ball"])
    if spec.get("fragment"):
        pxy, team = fragment_tracks(pxy, team, rng)
    if spec.get("vball"):
        # the scenario's flips are off (BASE), so the clean ball is in the same frame of reference
        bxy = vball_fill(bxy, m.ball_xy, pxy, np.random.default_rng([seed, k, 7]), **spec["vball"])
    return pxy, team, bxy


def model_probs(model_dir, m, pxy, team, bxy):
    if os.path.isfile(os.path.join(model_dir, "best.pt")):
        import torch
        from .infer import load_model, predict_match, window_of
        torch.set_num_threads(os.cpu_count() or 4)
        net, ck = load_model(os.path.join(model_dir, "best.pt"))
        return predict_match(net, pxy, team, bxy, m.runs(), m.fps, **window_of(ck))
    import pickle
    from .gbdt import predict_probs
    from .handfeat import match_features
    with open(os.path.join(model_dir, "model.pkl"), "rb") as fh:
        gbdt = pickle.load(fh)
    X, rows = match_features(pxy, team, bxy, m.runs(), m.fps)
    return predict_probs(gbdt, X, rows, m.n_frames)


CACHE = "outputs/stress_cache"


def cached_probs(model_dir, scenario, m, inp, seed=4321):
    """model_probs with a disk cache keyed by model, scenario spec, seed and match."""
    key = json.dumps([SCENARIOS[scenario], seed, BASE.__dict__], sort_keys=True, default=str)
    tag = hashlib.sha1(key.encode()).hexdigest()[:10]
    path = os.path.join(CACHE, model_dir.replace("/", "_").replace("\\", "_"),
                        f"{scenario}_{tag}_{m.name}.npz")
    ckpt = [os.path.join(model_dir, f) for f in ("best.pt", "model.pkl")
            if os.path.isfile(os.path.join(model_dir, f))][0]
    if os.path.isfile(path) and os.path.getmtime(path) > os.path.getmtime(ckpt):
        return np.load(path)["probs"]
    probs = model_probs(model_dir, m, *inp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, probs=probs)
    return probs


def spec_probs(spec, scenario, m, inp):
    dirs = spec.split("@")[0].split("+")
    return sum(cached_probs(d, scenario, m, inp).astype(np.float64) for d in dirs) / len(dirs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=["idsse_J03WMX", "idsse_J03WR9"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", default="outputs/stress.json")
    ap.add_argument("--vball", nargs="+", type=float, action="append",
                    metavar="FILL PREC ERR [BURST]",
                    help="add scenario atd_like + video-ball refill (repeatable)")
    args = ap.parse_args()
    for v in args.vball or []:
        fill, prec, err = v[:3]
        burst = v[3] if len(v) > 3 else 1.0
        name = f"vball_f{fill:g}_p{prec:g}_e{err:g}" + (f"_b{burst:g}" if burst > 1 else "")
        vb = dict(fill=fill, prec=prec, err=err)
        if burst > 1:
            vb["burst"] = burst            # key only when used: old scenario caches stay valid
        SCENARIOS[name] = {**SCENARIOS["atd_like"], "vball": vb}
        if name not in args.scenarios:
            args.scenarios.append(name)

    ms = [m for m in load_matches("cache/idsse_*.npz", 25) if m.name in args.val]
    decoders = {d: load_decoder(d.split("@")[1] if "@" in d else
                                os.path.join(d.split("+")[0], "decoder.json")) for d in args.models}
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
            for m, inp in zip(ms, inputs):
                codes = probs_to_codes(spec_probs(d, name, m, inp), m.runs(), **decoders[d])
                r = score(m.gt_codes(), codes)
                spells.append(r["spell_f1_mean"])
                poss.append(f"{r['n_pred_spells']}/{r['n_gt_spells']}")
            table[name][d] = float(np.mean(spells))
            print(f"{name:<14} {d:<20} spellF1 {np.mean(spells):.4f}  poss {' '.join(poss)}  "
                  f"(slots {n_slots}, ball seen {ball_seen:.0%}, {time.time() - t0:.0f}s)", flush=True)
    w = max(20, *(len(d) for d in args.models))
    print("\nscenario       " + "  ".join(f"{d:>{w}}" for d in args.models))
    for name, row in table.items():
        print(f"{name:<14} " + "  ".join(f"{row[d]:>{w}.4f}" for d in args.models))
    print(f"{'mean':<14} " + "  ".join(
        f"{np.mean([table[n][d] for n in table]):>{w}.4f}" for d in args.models))
    with open(args.out, "w") as fh:
        json.dump(table, fh, indent=1)


if __name__ == "__main__":
    main()
