"""IDSSE open data (Bassek et al., Sci Data 2025; CC BY 4.0) -> `Match` with H/A/D labels.

Seven full German matches (Bundesliga / 2. Bundesliga 2022-23), DFL positional data at 25 fps.
Every frame carries BallStatus (alive/dead) and BallPossession (home/away), i.e. exactly the two
things our ground truth is derived from -- "corrected tracking marks the ball dead / names a team".
These labels are from OTHER games, so training and validating on them is allowed (RULES.md).

Caveat worth remembering: DFL's possession semantics are not guaranteed to match Metrica's.
Check against the public 5 minutes how the competition GT treats short touches before trusting
external labels blindly.

Download (browser -- figshare blocks scripted downloads) into external/idsse/<match_id>/ as
meta.xml, tracking.xml and (optionally) event.xml. File ids are in MATCHES below:
    https://springernature.figshare.com/ndownloader/files/<file_id>

    python -m mtah.external.idsse            # convert every downloaded match to cache/*.npz
"""

import glob
import os
import time

import numpy as np

from ..common import CACHE_DIR, D, ROOT, X, Match

EXTERNAL_DIR = os.path.join(ROOT, "external", "idsse")

# match_id -> (description, {asset: figshare file id})
MATCHES = {
    "J03WMX": ("1. FC Koeln vs FC Bayern Muenchen",
               {"meta": 51643475, "event": 51643493, "tracking": 51643514}),
    "J03WN1": ("VfL Bochum vs Bayer 04 Leverkusen",
               {"meta": 51643472, "event": 51643496, "tracking": 51643517}),
    "J03WPY": ("Fortuna Duesseldorf vs 1. FC Nuernberg",
               {"meta": 51643487, "event": 51643505, "tracking": 51643526}),
    "J03WOH": ("Fortuna Duesseldorf vs SSV Jahn Regensburg",
               {"meta": 51643478, "event": 51643499, "tracking": 51643520}),
    "J03WQQ": ("Fortuna Duesseldorf vs FC St. Pauli",
               {"meta": 51643484, "event": 51643508, "tracking": 51643529}),
    "J03WOY": ("Fortuna Duesseldorf vs F.C. Hansa Rostock",
               {"meta": 51643481, "event": 51643502, "tracking": 51643523}),
    "J03WR9": ("Fortuna Duesseldorf vs 1. FC Kaiserslautern",
               {"meta": 51643490, "event": 51643511, "tracking": 51643532}),
}


def download_urls():
    for mid, (desc, files) in MATCHES.items():
        for asset in ("meta", "tracking", "event"):
            print(f"{mid}/{asset}.xml  https://springernature.figshare.com/ndownloader/files/"
                  f"{files[asset]}   # {desc}")


def convert(match_id, folder=None):
    """One IDSSE match as a Match. Labels: D where the ball is dead, else the owning team."""
    from kloppy import sportec
    from kloppy.domain import BallState

    folder = folder or os.path.join(EXTERNAL_DIR, match_id)
    t0 = time.time()
    ds = sportec.load_tracking(meta_data=os.path.join(folder, "meta.xml"),
                               raw_data=os.path.join(folder, "tracking.xml"),
                               only_alive=False)
    home, away = ds.metadata.teams
    players = list(home.players) + list(away.players)
    index = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)

    dims = ds.metadata.pitch_dimensions
    length = getattr(dims, "pitch_length", None) or 105.0
    width = getattr(dims, "pitch_width", None) or 68.0

    # Frames are listed per period with a gap between; we lay them out back to back, one period
    # after the other, with the period id kept per row so nothing is decoded across halftime.
    frames = ds.frames
    T, S = len(frames), len(players)
    xy = np.full((T, S, 2), np.nan, dtype=np.float32)
    speed = np.full((T, S), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    ball_z = np.full((T, 1), np.nan, dtype=np.float32)
    period = np.zeros(T, dtype=np.int8)
    label = np.full(T, X, dtype=np.int8)

    for t, fr in enumerate(frames):
        period[t] = fr.period.id
        if fr.ball_coordinates is not None:
            ball[t] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
            ball_z[t, 0] = fr.ball_coordinates.z if fr.ball_coordinates.z is not None else np.nan
        for p, pdata in fr.players_data.items():
            s = index.get(p.player_id)
            if s is None or pdata.coordinates is None:
                continue
            xy[t, s] = (pdata.coordinates.x, pdata.coordinates.y)
            speed[t, s] = pdata.speed if pdata.speed is not None else np.nan
        if fr.ball_state == BallState.DEAD:
            label[t] = D
        elif fr.ball_owning_team is not None:
            label[t] = 0 if fr.ball_owning_team.team_id == home.team_id else 1

    scale = np.array([length, width], dtype=np.float32)
    m = Match(name=f"idsse_{match_id}", players_xy=(xy - 0.5) * scale, team=team,
              ball_xy=(ball - 0.5) * scale, period=period, fps=int(ds.metadata.frame_rate or 25),
              pitch=(float(length), float(width)), players_speed=speed, ball_extra=ball_z,
              label=label, meta={"source": "idsse", "match_id": match_id,
                                 "desc": MATCHES.get(match_id, ("",))[0]})
    print(f"{m.name}: {T} frames, {S} players, {time.time() - t0:.0f}s; labels "
          + " ".join(f"{c}={int((label == k).sum())}" for k, c in enumerate("HADX")))
    return m


def load(match_id, refresh=False):
    path = os.path.join(CACHE_DIR, f"idsse_{match_id}.npz")
    if os.path.isfile(path) and not refresh:
        return Match.load(path)
    m = convert(match_id)
    m.save(path)
    return m


def available():
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(EXTERNAL_DIR, "*", "tracking.xml")))


if __name__ == "__main__":
    have = available()
    if not have:
        print("no IDSSE matches under", EXTERNAL_DIR, "-- download these in a browser:\n")
        download_urls()
    for mid in have:
        load(mid)
