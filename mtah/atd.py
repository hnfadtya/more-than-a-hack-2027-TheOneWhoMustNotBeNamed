"""Read the provided ATD (Metrica FIFA/EPTS) into a `Match`, cached as .npz.

Raw line format (see starter/baseline.py):  "<frame_id>:<slot>;<slot>;...:<ball>"
each group comma-separated, "x,y,speed" for players, NaN where missing. frame_id counts from 1,
the submission from 0, so row i of every array here is video frame i.

The number of fields per group is not hard-coded: it is read off the data, and every ball field
beyond (x, y) is kept raw in `ball_extra` so we can find out what it holds once the data arrives.

    python -m mtah.atd                 # parse, cache, print a summary
    python -m mtah.atd --source broadcast
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from .common import CACHE_DIR, ROOT, Match

sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

NAN = float("nan")


def _float(token):
    try:
        return float(token)
    except ValueError:
        return NAN


def team_of_each_track(meta_path):
    """0 for home, 1 for away, per track slot -- first <Team> in the XML is home (EPTS)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def parse_raw(raw_path, n_frames, n_slots):
    """(players (T,S,k) float32, ball (T,kb) float32) with rows indexed by video frame."""
    players = None
    ball = None
    with open(raw_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")
            ball_tokens = parts[2].split(",")

            if players is None:
                k = max(len(g.split(",")) for g in groups)
                players = np.full((n_frames, n_slots, k), np.nan, dtype=np.float32)
                ball = np.full((n_frames, len(ball_tokens)), np.nan, dtype=np.float32)

            for s, g in enumerate(groups[:n_slots]):
                if not g or g[0] == "N":
                    continue
                vals = g.split(",")
                players[row, s, :len(vals)] = [_float(v) for v in vals]
            ball[row, :len(ball_tokens)] = [_float(v) for v in ball_tokens[:ball.shape[1]]]
    return players, ball


def period_of_frames(n_frames, game=GAME):
    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in half_video_blocks(game):
        period[first:last + 1] = half
    return period


def build(game=GAME, source="tactical", data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    team = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]

    t0 = time.time()
    raw_players, raw_ball = parse_raw(raw_path, n_frames, len(team))
    print(f"parsed {os.path.basename(raw_path)} in {time.time() - t0:.0f}s: "
          f"player fields {raw_players.shape[2]}, ball fields {raw_ball.shape[1]}")

    # normalised 0-1 -> metres about the centre spot
    scale = np.array([length, width], dtype=np.float32)
    players_xy = (raw_players[..., :2] - 0.5) * scale
    ball_xy = (raw_ball[:, :2] - 0.5) * scale
    speed = raw_players[..., 2] if raw_players.shape[2] > 2 else None

    period = (period_of_frames(n_frames, game) if source == "tactical"
              else np.ones(n_frames, dtype=np.int8))  # broadcast has its own timeline
    return Match(name=f"{game}_{source}", players_xy=players_xy, team=team, ball_xy=ball_xy,
                 period=period, fps=GAMES[game]["fps"], pitch=(length, width),
                 players_speed=speed,
                 ball_extra=raw_ball[:, 2:] if raw_ball.shape[1] > 2 else None,
                 meta={"source": source, "game": game})


def load(game=GAME, source="tactical", data_dir=None, refresh=False):
    """The ATD as a Match, parsed once and then read from cache/<game>_<source>_atd.npz."""
    path = os.path.join(CACHE_DIR, f"{game}_{source}_atd.npz")
    if os.path.isfile(path) and not refresh:
        return Match.load(path)
    match = build(game, source, data_dir)
    match.save(path)
    return match


def summary(m):
    graded = m.period != 0
    ball_seen = ~np.isnan(m.ball_xy[:, 0])
    n_tracked = (~np.isnan(m.players_xy[..., 0])).sum(axis=1)
    print(f"{m.name}: {m.n_frames} frames, {len(m.team)} slots "
          f"(home {int((m.team == 0).sum())}, away {int((m.team == 1).sum())})")
    print(f"  ball seen: all {ball_seen.mean():.1%}  graded {ball_seen[graded].mean():.1%}")
    print(f"  tracked objects per graded frame: mean {n_tracked[graded].mean():.1f}, "
          f"max {n_tracked[graded].max()}")
    if m.ball_extra is not None:
        for j in range(m.ball_extra.shape[1]):
            col = m.ball_extra[ball_seen, j]
            col = col[~np.isnan(col)]
            if len(col):
                # aggregates only: no raw values leave the machine
                print(f"  ball extra field {j}: n={len(col)} unique={len(np.unique(col))} "
                      f"min={col.min():.3g} max={col.max():.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    summary(load(source=args.source, refresh=args.refresh))
