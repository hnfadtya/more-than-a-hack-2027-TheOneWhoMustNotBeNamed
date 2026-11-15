"""Ball detection from the tactical video, to fill the ATD's ball gaps (Jalur 1).

The ATD loses the ball on ~41% of graded frames, in long gaps (median ~11 s). A per-frame detector
on the video, mapped to the pitch with the provided homography, can recover part of that.

Validation needs no labels: on frames where the ATD does see the ball, a detection either lands
on it or it does not. Everything printed here is an aggregate; no frame, position or image leaves
the machine.

    python -m mtah.vball probe --weights yolo11s.pt --imgsz 1920
"""

import argparse
import os
import sys
import time

import numpy as np

from .common import ROOT, Match

sys.path.insert(0, ROOT)
from games import GAME, GAMES, video_path  # noqa: E402

COCO_SPORTS_BALL = 32
FRAMES_PER_BLOCK = 25 * 60          # one minute of video
VAL_EVERY = 10                      # every 10th minute is held out for validation
TILE = 640


def is_val(frames):
    """Held-out time blocks: every VAL_EVERY-th minute of video. Never used for training."""
    return (np.asarray(frames) // FRAMES_PER_BLOCK) % VAL_EVERY == VAL_EVERY - 1


def load_homography(game=GAME):
    from starter.load_homography import load
    mats, err, valid = load(game)
    return mats, err, valid


def pitch_to_pixel(mat, xy_m, pitch):
    """metres about the centre spot (N,2) -> pixels (N,2)."""
    L, W = pitch
    u = xy_m[:, 0] / L + 0.5
    v = xy_m[:, 1] / W + 0.5
    den = mat[2, 0] * u + mat[2, 1] * v + mat[2, 2]
    px = (mat[0, 0] * u + mat[0, 1] * v + mat[0, 2]) / den
    py = (mat[1, 0] * u + mat[1, 1] * v + mat[1, 2]) / den
    return np.stack([px, py], axis=1)


def pixel_to_pitch(mat, pix, pitch):
    """pixels (N,2) -> metres about the centre spot (N,2), assuming the point is on the ground."""
    L, W = pitch
    inv = np.linalg.inv(mat)
    x, y = pix[:, 0], pix[:, 1]
    den = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2]
    u = (inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]) / den
    v = (inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]) / den
    return np.stack([(u - 0.5) * L, (v - 0.5) * W], axis=1)


def read_frames(path, start, n):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def detect(model, frames, imgsz, conf, device):
    """-> list per frame of (K,3) arrays: pixel x, pixel y (box centre), confidence."""
    # COCO models: class 32 'sports ball'; our fine-tuned models have a single class 0 'ball'
    cls = 0 if len(model.names) == 1 else COCO_SPORTS_BALL
    res = model.predict(frames, imgsz=imgsz, conf=conf, classes=[cls],
                        device=device, verbose=False)
    out = []
    for r in res:
        b = r.boxes
        if b is None or len(b) == 0:
            out.append(np.zeros((0, 3)))
            continue
        xywh = b.xywh.cpu().numpy()
        c = b.conf.cpu().numpy()
        out.append(np.column_stack([xywh[:, 0], xywh[:, 1], c]))
    return out


def probe(args):
    from ultralytics import YOLO

    m = Match.load(os.path.join(ROOT, "cache", f"{GAME}_tactical_atd.npz"))
    pitch = GAMES[GAME]["pitch"]
    mats, _, valid = load_homography()
    graded = m.period != 0
    vis = ~np.isnan(m.ball_xy[:, 0])

    rng = np.random.default_rng(args.seed)
    ok = graded.copy()
    if args.split == "val":
        ok &= is_val(np.arange(m.n_frames))
    cand = np.flatnonzero(ok[:-args.seg_len])
    starts = np.sort(rng.choice(cand, size=args.n_seg, replace=False))

    model = YOLO(args.weights)
    path = video_path()
    dets, frames_idx = [], []
    t_read = t_det = 0.0
    for s in starts:
        t0 = time.time()
        frs = read_frames(path, int(s), args.seg_len)
        t1 = time.time()
        for k in range(0, len(frs), args.batch):
            dets += detect(model, frs[k:k + args.batch], args.imgsz, args.conf, args.device)
        t2 = time.time()
        frames_idx += list(range(int(s), int(s) + len(frs)))
        t_read += t1 - t0
        t_det += t2 - t1
    frames_idx = np.array(frames_idx)
    n = len(frames_idx)
    print(f"{args.weights} imgsz {args.imgsz}: {n} frames, detect {n / t_det:.1f} fps, "
          f"read+seek {n / t_read:.1f} fps")

    # candidates in pitch metres
    cand_m = []
    for f, d in zip(frames_idx, dets):
        if len(d) and valid[f]:
            cand_m.append(np.column_stack([pixel_to_pitch(mats[f], d[:, :2], pitch), d[:, 2]]))
        else:
            cand_m.append(np.zeros((0, 3)))

    def dist_to_atd(j, f):
        c = cand_m[j]
        if not len(c) or not vis[f]:
            return np.full(len(c), np.inf)
        return np.linalg.norm(c[:, :2] - m.ball_xy[f], axis=1)

    v = vis[frames_idx]
    print(f"  sampled frames with ATD ball visible: {v.mean():.1%}")
    for tau in args.taus:
        has = np.array([bool(len(c)) and c[:, 2].max() >= tau for c in cand_m])
        top_ok, any_ok, top_err = [], [], []
        for j, f in enumerate(frames_idx):
            if not v[j]:
                continue
            d = dist_to_atd(j, f)
            keep = cand_m[j][:, 2] >= tau
            if keep.any():
                dk = d[keep]
                top = dk[np.argmax(cand_m[j][keep, 2])]
                top_ok.append(top < args.radius)
                any_ok.append(dk.min() < args.radius)
                if top < args.radius:
                    top_err.append(top)
            else:
                any_ok.append(False)
        top_ok, any_ok = np.array(top_ok), np.array(any_ok)
        print(f"  conf>={tau:.2f}: [ATD visible] detected {has[v].mean():.1%}, "
              f"recall(any<{args.radius}m) {any_ok.mean():.1%}, "
              f"top1 precision {top_ok.mean() if len(top_ok) else float('nan'):.1%}, "
              f"top1 err median {np.median(top_err) if top_err else float('nan'):.2f} m | "
              f"[ATD missing] detected {has[~v].mean() if (~v).any() else float('nan'):.1%}")

    # frame-alignment check: top1 vs ATD ball at t+k should peak at k = 0
    tau = args.taus[0]
    line = []
    for k in range(-3, 4):
        ok = []
        for j, f in enumerate(frames_idx):
            g = f + k
            c = cand_m[j]
            if not len(c) or c[:, 2].max() < tau or not (0 <= g < m.n_frames) or not vis[g]:
                continue
            top = c[np.argmax(c[:, 2]), :2]
            ok.append(np.linalg.norm(top - m.ball_xy[g]) < args.radius)
        line.append(f"{k:+d}:{np.mean(ok) if ok else float('nan'):.1%}")
    print("  alignment (top1 hit vs ATD at t+k): " + "  ".join(line))


def fill_path(tag, game=GAME):
    return os.path.join(ROOT, "cache", f"{game}_vball_{tag}.npz")


def fill(args):
    """Detect the ball on every graded frame where the ATD has none -> cache/<game>_vball_<tag>.npz.

    Exactly what Stage A measured: top-1 detection per frame at conf >= tau, mapped to the pitch
    with that frame's homography, no further filtering. Positions are rounded to 1 cm and
    confidences to 1e-3 so tiny numeric differences between devices do not leak further.
    Saves a checkpoint every --save_every frames and resumes from it.
    """
    import cv2
    from ultralytics import YOLO

    m = Match.load(os.path.join(ROOT, "cache", f"{GAME}_tactical_atd.npz"))
    pitch = GAMES[GAME]["pitch"]
    mats, _, valid = load_homography()
    need = np.flatnonzero((m.period != 0) & np.isnan(m.ball_xy[:, 0]) & valid)
    if args.limit:
        need = need[:args.limit]                      # smoke tests only
    out = fill_path(args.tag)
    part = out + ".partial.npz"
    done_f, done_xy, done_c = [], [], []
    start = 0
    if os.path.isfile(part):
        z = np.load(part)
        start = int(z["next"])
        done_f, done_xy, done_c = list(z["frame"]), list(z["xy"]), list(z["conf"])
        print(f"resuming at {start}/{len(need)}")
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(video_path())
    todo = need[start:]
    pos = 0
    if len(todo):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(todo[0]))
        pos = int(todo[0])
    t0 = time.time()
    buf_f, buf_img = [], []

    def flush():
        if not buf_f:
            return
        dets = detect(model, buf_img, args.imgsz, args.tau, args.device)
        for f, d in zip(buf_f, dets):
            if not len(d):
                continue
            top = d[np.argmax(d[:, 2])]
            xy = pixel_to_pitch(mats[f], top[None, :2], pitch)[0]
            done_f.append(f)
            done_xy.append(np.round(xy, 2).astype(np.float32))
            done_c.append(np.float32(round(float(top[2]), 3)))
        buf_f.clear()
        buf_img.clear()

    def save(path, nxt):
        np.savez_compressed(path, frame=np.array(done_f, dtype=np.int32),
                            xy=np.array(done_xy, dtype=np.float32).reshape(-1, 2),
                            conf=np.array(done_c, dtype=np.float32), next=nxt,
                            weights=os.path.basename(args.weights), tau=args.tau,
                            imgsz=args.imgsz, device=args.device)

    for i, f in enumerate(todo):
        while pos < f:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            raise SystemExit(f"video ended early at frame {f}")
        buf_f.append(int(f))
        buf_img.append(img)
        if len(buf_f) == args.batch:
            flush()
        k = start + i + 1
        if k % args.save_every == 0:
            flush()
            save(part, k)
            rate = (i + 1) / (time.time() - t0)
            print(f"  {k}/{len(need)} frames, {rate:.1f} fps, eta {(len(need) - k) / rate / 60:.0f} min,"
                  f" detected {len(done_f) / k:.1%}", flush=True)
    if buf_f:
        flush()
    cap.release()
    save(out, len(need))
    if os.path.isfile(part):
        os.remove(part)
    print(f"wrote {out}: {len(done_f)} detections on {len(need)} gap frames "
          f"({len(done_f) / max(1, len(need)):.1%})")


def apply_fill(m, tag):
    """Copy of the ATD ball with the gaps filled from cache/<game>_vball_<tag>.npz."""
    z = np.load(fill_path(tag))
    ball = m.ball_xy.copy()
    f = z["frame"]
    gap = np.isnan(ball[f, 0])            # never overwrite an ATD ball
    ball[f[gap]] = z["xy"][gap]
    return ball, int(gap.sum())


def px_per_metre(mat, xy_m, pitch):
    """Local image scale (pixels per metre) around each pitch point (N,2)."""
    c = pitch_to_pixel(mat, xy_m, pitch)
    dx = pitch_to_pixel(mat, xy_m + [0.5, 0.0], pitch) - c
    dy = pitch_to_pixel(mat, xy_m + [0.0, 0.5], pitch) - c
    return (np.linalg.norm(dx, axis=1) + np.linalg.norm(dy, axis=1))   # mean of 2 x (0.5 m)


def dataset(args):
    """YOLO dataset of 640x640 tiles at native resolution, boxes from the ATD ball.

    Positive tile: contains the ATD ball (at a random place in the tile). Negative tile: from the
    same frame, away from the ATD ball (so spare balls and look-alikes are taught as background).
    Train frames and val frames come from disjoint time blocks (`is_val`).
    """
    import cv2

    m = Match.load(os.path.join(ROOT, "cache", f"{GAME}_tactical_atd.npz"))
    pitch = GAMES[GAME]["pitch"]
    mats, _, valid = load_homography()
    W, Hh = 1920, 1080
    graded = m.period != 0
    vis = ~np.isnan(m.ball_xy[:, 0])
    ok = graded & vis & valid
    frames = np.flatnonzero(ok)

    # ball pixel and box size for every candidate frame
    pix = np.full((m.n_frames, 2), np.nan)
    size = np.full(m.n_frames, np.nan)
    for f in frames:
        pix[f] = pitch_to_pixel(mats[f], m.ball_xy[f:f + 1], pitch)[0]
        size[f] = np.clip(args.box_m * px_per_metre(mats[f], m.ball_xy[f:f + 1], pitch)[0],
                          args.box_min, args.box_max)
    inside = (pix[frames, 0] > 4) & (pix[frames, 0] < W - 4) & (pix[frames, 1] > 4) & \
             (pix[frames, 1] < Hh - 4)
    frames = frames[inside]

    rng = np.random.default_rng(args.seed)
    val_f = frames[is_val(frames)]
    tr_f = frames[~is_val(frames)]
    # spread the sample: at least args.gap frames apart (consecutive frames are near duplicates)
    def spread(fs, n):
        fs = fs[fs % args.gap == 0] if args.gap > 1 else fs
        return np.sort(rng.choice(fs, size=min(n, len(fs)), replace=False))
    sel = {"train": spread(tr_f, args.n_train), "val": spread(val_f, args.n_val)}
    split_of = {int(f): s for s, fs in sel.items() for f in fs}
    print(f"candidate frames: train {len(tr_f)}, val {len(val_f)} -> sampled "
          f"train {len(sel['train'])}, val {len(sel['val'])}")
    print(f"box size px: median {np.median(size[frames]):.1f}, "
          f"p10 {np.percentile(size[frames], 10):.1f}, p90 {np.percentile(size[frames], 90):.1f}")

    out = args.out
    for s in sel:
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)

    def write(split, name, img, boxes):
        cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
        with open(os.path.join(out, "labels", split, name + ".txt"), "w") as fh:
            for cx, cy, w in boxes:
                fh.write(f"0 {cx / TILE:.6f} {cy / TILE:.6f} {w / TILE:.6f} {w / TILE:.6f}\n")

    cap = cv2.VideoCapture(video_path())
    want = sorted(split_of)
    last = want[-1]
    t0 = time.time()
    n_pos = n_neg = 0
    j = 0
    for f in range(last + 1):
        if f != want[j]:
            cap.grab()
            continue
        ok_, img = cap.read()
        j += 1
        if not ok_:
            break
        split = split_of[f]
        bx, by = pix[f]
        w = size[f]
        # positive tile: ball at a random place, at least 16 px from the edge
        x0 = int(np.clip(rng.uniform(bx - TILE + 16, bx - 16), 0, W - TILE))
        y0 = int(np.clip(rng.uniform(by - TILE + 16, by - 16), 0, Hh - TILE))
        write(split, f"f{f:06d}_p", img[y0:y0 + TILE, x0:x0 + TILE], [(bx - x0, by - y0, w)])
        n_pos += 1
        if rng.random() < args.neg_frac:
            for _ in range(20):
                x0 = int(rng.integers(0, W - TILE + 1))
                y0 = int(rng.integers(0, Hh - TILE + 1))
                if not (x0 - 32 <= bx <= x0 + TILE + 32 and y0 - 32 <= by <= y0 + TILE + 32):
                    write(split, f"f{f:06d}_n", img[y0:y0 + TILE, x0:x0 + TILE], [])
                    n_neg += 1
                    break
        if j % 1000 == 0:
            print(f"  {j}/{len(want)} frames, {f / (time.time() - t0):.0f} video fps")
        if j == len(want):
            break
    cap.release()
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    print(f"wrote {n_pos} positive and {n_neg} negative tiles to {out} "
          f"in {time.time() - t0:.0f}s")


def refine(args):
    """Fix the ATD-derived boxes: the ATD marks the ground point under the ball (measured: the
    ball centre sits ~1 radius above it) with ~0.3-0.5 m noise, too much for a 12 px box.

    Per positive tile: if the COCO-pretrained detector finds a ball within `snap_px` of the label,
    move the label onto it; otherwise shift the label up by one ball radius. Box side becomes
    `box_m` metres at the local scale. The original labels are kept in labels_atd/.
    """
    import shutil
    from ultralytics import YOLO

    model = YOLO(args.weights)
    n_snap = n_all = n_drop = 0
    shifts = []
    for split in ("train", "val"):
        lab_dir = os.path.join(args.dataset, "labels", split)
        bak_dir = os.path.join(args.dataset, "labels_atd", split)
        if not os.path.isdir(bak_dir):
            shutil.copytree(lab_dir, bak_dir)
        img_dir = os.path.join(args.dataset, "images", split)
        names = sorted(n for n in os.listdir(img_dir) if n.endswith("_p.jpg"))
        for i in range(0, len(names), args.batch):
            chunk = names[i:i + args.batch]
            dets = detect(model, [os.path.join(img_dir, n) for n in chunk], TILE, args.conf,
                          args.device)
            for n, d in zip(chunk, dets):
                with open(os.path.join(bak_dir, n[:-4] + ".txt")) as fh:
                    _, cx, cy, w, _ = map(float, fh.read().split())
                cx, cy, w = cx * TILE, cy * TILE, w * TILE
                scale = w / 0.6                                     # px per metre (dataset box_m)
                n_all += 1
                near = np.linalg.norm(d[:, :2] - [cx, cy], axis=1) < args.snap_px if len(d) \
                    else np.zeros(0, bool)
                if near.any():
                    k = np.flatnonzero(near)[np.argmax(d[near, 2])]
                    nx, ny = d[k, 0], d[k, 1]
                    n_snap += 1
                    shifts.append((nx - cx, ny - cy))
                elif args.drop_unsnapped:
                    # ATD alone is wrong on most such frames (human check: 12/15 off). Drop the
                    # positive AND the frame's negative: the real ball may sit in that negative.
                    stem = n[:-len("_p.jpg")]
                    for suf in ("_p", "_n"):
                        for p in (os.path.join(img_dir, stem + suf + ".jpg"),
                                  os.path.join(lab_dir, stem + suf + ".txt")):
                            if os.path.isfile(p):
                                os.remove(p)
                    n_drop += 1
                    continue
                else:
                    nx, ny = cx, cy - 0.11 * scale
                nw = float(np.clip(args.box_m * scale, args.box_min, args.box_max))
                with open(os.path.join(lab_dir, n[:-4] + ".txt"), "w") as fh:
                    fh.write(f"0 {nx / TILE:.6f} {ny / TILE:.6f} {nw / TILE:.6f} {nw / TILE:.6f}\n")
    s = np.array(shifts)
    print(f"refined {n_all} positive tiles: snapped to a detection {n_snap / n_all:.1%}, "
          f"snap shift median dx {np.median(s[:, 0]):+.1f} dy {np.median(s[:, 1]):+.1f} px, "
          f"|shift| median {np.median(np.linalg.norm(s, axis=1)):.1f} px; "
          f"dropped {n_drop} unsnapped frames")
    for split in ("train", "val"):
        imgs = os.listdir(os.path.join(args.dataset, "images", split))
        print(f"  {split}: {sum(i.endswith('_p.jpg') for i in imgs)} positive, "
              f"{sum(i.endswith('_n.jpg') for i in imgs)} negative tiles")


def check(args):
    """Draw the label boxes on a few positive tiles, for a human to eyeball (stays local)."""
    import cv2
    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.dataset, "images", "train")
    names = sorted(n for n in os.listdir(src) if n.endswith("_p.jpg"))
    os.makedirs(args.out, exist_ok=True)
    for n in rng.choice(names, size=min(args.n, len(names)), replace=False):
        img = cv2.imread(os.path.join(src, n))
        lab = os.path.join(args.dataset, "labels", "train", n[:-4] + ".txt")
        for line in open(lab):
            _, cx, cy, w, h = map(float, line.split())
            x0, y0 = int((cx - w / 2) * TILE), int((cy - h / 2) * TILE)
            x1, y1 = int((cx + w / 2) * TILE), int((cy + h / 2) * TILE)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        atd = os.path.join(args.dataset, "labels_atd", "train", n[:-4] + ".txt")
        if os.path.isfile(atd):          # original ATD point (before refine) as a small blue dot
            _, cx, cy, _, _ = map(float, open(atd).read().split())
            cv2.circle(img, (int(cx * TILE), int(cy * TILE)), 2, (255, 0, 0), -1)
        if args.zoom:   # 4x crop around the label, so a 4 px ball is visible to the eye
            _, cx, cy, _, _ = map(float, open(lab).read().split())
            c = 64
            x0 = int(np.clip(cx * TILE - c, 0, TILE - 2 * c))
            y0 = int(np.clip(cy * TILE - c, 0, TILE - 2 * c))
            img = cv2.resize(img[y0:y0 + 2 * c, x0:x0 + 2 * c], None, fx=args.zoom, fy=args.zoom,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.out, n), img)
    print(f"wrote {min(args.n, len(names))} tiles with boxes to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fl = sub.add_parser("fill")
    fl.add_argument("--weights", default=os.path.join(ROOT, "outputs", "vball", "ft1", "best.pt"))
    fl.add_argument("--tag", default="ft1")
    fl.add_argument("--tau", type=float, default=0.05)     # Stage A choice, WORKLOG 2026-11-15
    fl.add_argument("--imgsz", type=int, default=1920)
    fl.add_argument("--batch", type=int, default=8)
    fl.add_argument("--save_every", type=int, default=2000)
    fl.add_argument("--device", default="mps")
    fl.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("refine")
    r.add_argument("--dataset", default=os.path.join(ROOT, "outputs", "vball", "dataset"))
    r.add_argument("--weights", default=os.path.join(ROOT, "outputs", "vball", "yolo11s.pt"))
    r.add_argument("--snap_px", type=float, default=15.0)
    r.add_argument("--conf", type=float, default=0.03)
    r.add_argument("--box_m", type=float, default=1.0)
    r.add_argument("--box_min", type=float, default=12.0)
    r.add_argument("--box_max", type=float, default=40.0)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--device", default="mps")
    r.add_argument("--drop_unsnapped", action="store_true",
                   help="keep only frames where the detector confirms the ATD ball")
    c = sub.add_parser("check")
    c.add_argument("--dataset", default=os.path.join(ROOT, "outputs", "vball", "dataset"))
    c.add_argument("--out", default=os.path.join(ROOT, "outputs", "vball", "check"))
    c.add_argument("--n", type=int, default=30)
    c.add_argument("--zoom", type=int, default=0, help="e.g. 4: save a 4x crop around the label")
    c.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("dataset")
    d.add_argument("--out", default=os.path.join(ROOT, "outputs", "vball", "dataset"))
    d.add_argument("--n_train", type=int, default=6000)
    d.add_argument("--n_val", type=int, default=600)
    d.add_argument("--gap", type=int, default=5)
    d.add_argument("--neg_frac", type=float, default=0.5)
    d.add_argument("--box_m", type=float, default=0.6)
    d.add_argument("--box_min", type=float, default=8.0)
    d.add_argument("--box_max", type=float, default=32.0)
    d.add_argument("--jpeg", type=int, default=92)
    d.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("probe")
    p.add_argument("--split", default="all", choices=["all", "val"])
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--imgsz", type=int, default=1920)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.15, 0.3])
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--n_seg", type=int, default=20)
    p.add_argument("--seg_len", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "probe":
        probe(args)
    elif args.cmd == "dataset":
        dataset(args)
    elif args.cmd == "check":
        check(args)
    elif args.cmd == "refine":
        refine(args)
    elif args.cmd == "fill":
        fill(args)


if __name__ == "__main__":
    main()
