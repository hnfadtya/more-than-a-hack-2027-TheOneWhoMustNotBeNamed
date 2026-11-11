"""Run a trained PossessionNet over a whole match -> per-frame H/A/D probabilities.

Used by training (validation) and for the competition match. For the competition match this runs
locally on CPU: the data never has to leave the machine, and CPU inference with fixed threads is
what makes "the repository regenerates the exact file" achievable. Probabilities are rounded
before decoding so that last-bit float differences between machines cannot flip a decision.

    python -m mtah.infer --ckpt outputs/<run>/best.pt                       # BAR-ATH tactical
    python -m mtah.infer --ckpt outputs/<run>/best.pt --submission outputs/<run>/sub.csv
"""

import argparse
import os

import numpy as np
import torch

from .common import ROOT, X
from .decode import DEFAULT_DECODE, PROB_DECIMALS, probs_to_codes  # noqa: F401 (re-exported)
from .features import featurize
from .model import PossessionNet



def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = PossessionNet(**ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict_block(model, players, pmask, ball, L=750, hop=375, device="cpu", batch=16):
    """Probabilities (T,3) for one contiguous block, sliding overlapping windows.

    Overlapping predictions are blended with a triangular weight that favours window centres,
    where the bidirectional model has context on both sides.
    """
    T = len(ball)
    if T == 0:
        return np.zeros((0, 3), dtype=np.float32)
    Lw = min(L, T)
    starts = list(range(0, T - Lw + 1, hop))
    if starts[-1] + Lw < T:
        starts.append(T - Lw)
    weight = np.minimum(np.arange(Lw) + 1, Lw - np.arange(Lw)).astype(np.float64)
    acc = np.zeros((T, 3))
    wsum = np.zeros(T)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        P = torch.from_numpy(np.stack([players[s:s + Lw] for s in chunk])).to(device)
        M = torch.from_numpy(np.stack([pmask[s:s + Lw] for s in chunk])).to(device)
        Bl = torch.from_numpy(np.stack([ball[s:s + Lw] for s in chunk])).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(P, M, Bl)
        prob = torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
        for j, s in enumerate(chunk):
            acc[s:s + Lw] += prob[j] * weight[:, None]
            wsum[s:s + Lw] += weight
    return (acc / wsum[:, None]).astype(np.float32)


def predict_match(model, players_xy, team, ball_xy, runs, fps=25, device="cpu", **kw):
    """Probabilities (T,3) for a whole match, each run (half) predicted on its own. NaN outside."""
    T = len(ball_xy)
    out = np.full((T, 3), np.nan, dtype=np.float32)
    for a, b in runs:
        players, pmask, ball = featurize(players_xy[a:b], team, ball_xy[a:b], fps)
        out[a:b] = predict_block(model, players, pmask, ball, device=device, **kw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", help="probabilities .npz (default: next to the checkpoint)")
    ap.add_argument("--submission", help="also decode with default params and write a CSV here")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    from . import atd
    m = atd.load()
    model, ck = load_model(args.ckpt)
    runs = m.runs()
    probs = predict_match(model, m.players_xy, m.team, m.ball_xy, runs, m.fps)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"{m.name}_probs.npz")
    np.savez_compressed(out, probs=probs)
    print("wrote", os.path.relpath(out, ROOT))

    if args.submission:
        from .common import write_submission
        codes = probs_to_codes(probs, runs, **DEFAULT_DECODE)
        codes[m.period == 0] = X
        write_submission(args.submission, codes)
        print("wrote", args.submission, "-- validate it: python evaluation/validate_submission.py",
              args.submission)


if __name__ == "__main__":
    main()
