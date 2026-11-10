"""Train PossessionNet on labelled tracking from OTHER games (IDSSE), corrupted to look like ATD.

Validation is leave-matches-out and is scored the way the board scores: probabilities -> Viterbi
decoder -> official spellF1. The checkpoint kept as best.pt is the one with the best validation
spellF1, never anything measured on BAR-ATH.

    # smoke test, no data needed (CPU is fine)
    python -m mtah.train --synthetic 4 --epochs 2 --steps 20 --batch 4 --out outputs/smoke

    # real run (Colab A100), resumable: re-run the same command after a disconnect
    python -m mtah.train --data "cache/idsse_*.npz" --val idsse_J03WR9 --out outputs/run1
"""

import argparse
import glob
import json
import math
import os
import time
from dataclasses import fields

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .augment import CorruptCfg, corrupt
from .common import X, Match
from .features import featurize
from .infer import DEFAULT_DECODE, predict_match, probs_to_codes
from .labels import absorb_short_spells
from .model import PossessionNet
from .scoring import brief, score


# -- data ----------------------------------------------------------------------------------------

def load_matches(pattern, synthetic=0):
    if synthetic:
        from .synth import simulate
        return [simulate(f"synth{i}", minutes=20, seed=i) for i in range(synthetic)]
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no matches match {pattern!r}")
    return [Match.load(p) for p in paths]


def ball_z(m):
    return m.ball_extra[:, :1] if m.ball_extra is not None and m.ball_extra.shape[1] else None


class Windows(Dataset):
    """Random L-frame windows inside a run, freshly corrupted every time they are drawn."""

    def __init__(self, matches, L, n, cfg, seed):
        self.matches, self.L, self.n, self.cfg, self.seed = matches, L, n, cfg, seed
        self.blocks = [(i, a, b) for i, m in enumerate(matches) for a, b in m.runs() if b - a > 50]
        w = np.array([b - a for _, a, b in self.blocks], dtype=np.float64)
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        mi, a, b = self.blocks[rng.choice(len(self.blocks), p=self.p)]
        m = self.matches[mi]
        L = self.L
        s = int(rng.integers(a, max(a, b - L) + 1))
        e = min(b, s + L)
        z = ball_z(m)
        pxy, team, bxy, lab = corrupt(m.players_xy[s:e], m.team, m.ball_xy[s:e], m.label[s:e],
                                      rng, self.cfg, None if z is None else z[s:e])
        players, pmask, ball = featurize(pxy, team, bxy, m.fps)
        y = np.full(L, X, dtype=np.int64)
        y[:e - s] = lab
        pad = L - (e - s)
        if pad:
            players = np.concatenate([players, np.zeros((pad,) + players.shape[1:], np.float32)])
            pmask = np.concatenate([pmask, np.zeros((pad,) + pmask.shape[1:], bool)])
            ball = np.concatenate([ball, np.zeros((pad,) + ball.shape[1:], np.float32)])
        return (torch.from_numpy(players), torch.from_numpy(pmask), torch.from_numpy(ball),
                torch.from_numpy(y))


def class_weights(matches):
    counts = np.zeros(3)
    for m in matches:
        lab = m.label[m.period != 0]
        counts += np.bincount(lab[lab < 3], minlength=3)[:3]
    w = 1.0 / np.sqrt(np.maximum(counts, 1) / counts.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# -- validation ----------------------------------------------------------------------------------

def make_val_set(matches, cfg, seed=1234):
    """Each validation match corrupted ONCE with a fixed seed, so epochs are comparable."""
    out = []
    for k, m in enumerate(matches):
        rng = np.random.default_rng([seed, k])
        vcfg = CorruptCfg(**{**cfg.to_dict(), "flip": False, "team_swap": False})
        pxy, team, bxy, _ = corrupt(m.players_xy, m.team, m.ball_xy, None, rng, vcfg, ball_z(m))
        out.append((m, pxy, team, bxy))
    return out


def validate(model, val_set, device, L, decode=DEFAULT_DECODE):
    model.eval()
    results, all_probs = {}, {}
    for m, pxy, team, bxy in val_set:
        runs = m.runs()
        probs = predict_match(model, pxy, team, bxy, runs, m.fps, device=device, L=L, hop=L // 2)
        codes = probs_to_codes(probs, runs, **decode)
        gt = m.gt_codes()
        res = score(gt, codes)
        graded = gt < 3
        res["frame_acc_raw"] = float((np.nanargmax(np.nan_to_num(probs), 1)[graded]
                                      == gt[graded]).mean())
        results[m.name], all_probs[m.name] = res, probs
    model.train()
    return results, all_probs


# -- training ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="cache/idsse_*.npz")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic matches instead")
    ap.add_argument("--val", nargs="*", help="match names held out (default: the last one)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500, help="optimizer steps per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--L", type=int, default=750, help="window length in frames (30 s)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    ap.add_argument("--min_spell", type=int, default=25,
                    help="frames; shorter live-play team spells are uncontrolled touches and are "
                         "given back to the previous team (mtah.labels). 0 = raw labels")
    for f in fields(CorruptCfg):  # every corruption knob is a flag, e.g. --ball_missing 0.45
        if f.type in (float, "float"):
            ap.add_argument(f"--{f.name}", type=float, default=f.default)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = CorruptCfg(**{f.name: getattr(args, f.name) for f in fields(CorruptCfg)
                        if hasattr(args, f.name)})

    matches = load_matches(args.data, args.synthetic)
    for m in matches:  # same definition of possession for training targets and validation truth
        m.label = absorb_short_spells(m.label, m.runs(), args.min_spell)
    names = [m.name for m in matches]
    val_names = args.val or [names[-1]]
    train_m = [m for m in matches if m.name not in val_names]
    val_m = [m for m in matches if m.name in val_names]
    if not train_m or not val_m:
        raise SystemExit(f"need both train and val matches; have {names}, val {val_names}")
    print(f"train {[m.name for m in train_m]}\nval   {[m.name for m in val_m]}\ndevice {args.device}")

    model = PossessionNet(d=args.d).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * args.steps
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_m).to(args.device), ignore_index=X,
                                  label_smoothing=0.05)

    start_epoch, best, history = 0, -1.0, []
    last_path = os.path.join(args.out, "last.pt")
    if os.path.isfile(last_path) and not args.fresh:
        ck = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"resumed from epoch {ck['epoch']} (best spellF1 {best:.4f})")

    val_set = make_val_set(val_m, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {n_params / 1e6:.2f}M params, {args.steps} steps x {args.epochs} epochs")

    def checkpoint(path, epoch):
        torch.save({"model": model.state_dict(), "model_config": model.config, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best, "history": history,
                    "corrupt": cfg.to_dict(), "args": vars(args), "val": val_names}, path)

    use_amp = args.device != "cpu"
    for epoch in range(start_epoch, args.epochs):
        ds = Windows(train_m, args.L, args.steps * args.batch, cfg, seed=args.seed * 100003 + epoch)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=use_amp, drop_last=True)
        t0, run_loss = time.time(), 0.0
        model.train()
        for step, (P, M, B, y) in enumerate(dl):
            P, M, B, y = (t.to(args.device, non_blocking=True) for t in (P, M, B, y))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(P, M, B)
            loss = loss_fn(logits.float().reshape(-1, 3), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
        train_loss = run_loss / max(1, step + 1)

        results, probs = validate(model, val_set, args.device, args.L)
        spell = float(np.mean([r["spell_f1_mean"] for r in results.values()]))
        macro = float(np.mean([r["macro_f1"] for r in results.values()]))
        acc = float(np.mean([r["frame_acc_raw"] for r in results.values()]))
        history.append({"epoch": epoch, "loss": train_loss, "val_spellF1": spell,
                        "val_macroF1": macro, "val_frame_acc": acc, "secs": time.time() - t0})
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  val spellF1 {spell:.4f}  macroF1 {macro:.4f}"
              f"  acc {acc:.3f}  ({time.time() - t0:.0f}s)")
        for name, r in results.items():
            print(f"    {name:<20} {brief(r)}")

        if spell > best:
            best = spell
            checkpoint(os.path.join(args.out, "best.pt"), epoch)
            np.savez_compressed(os.path.join(args.out, "val_probs.npz"), **probs)
        checkpoint(last_path, epoch)
        with open(os.path.join(args.out, "history.json"), "w") as fh:
            json.dump(history, fh, indent=1)
    print(f"done. best val spellF1 {best:.4f} -> {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
