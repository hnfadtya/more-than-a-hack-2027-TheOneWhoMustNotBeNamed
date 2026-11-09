"""The one match representation every part of the pipeline speaks.

Competition ATD and external datasets (IDSSE / DFL, ...) are all converted to `Match`, so features,
models and the decoder never need to know where a frame came from.

Conventions
-----------
* coordinates in metres, origin at the centre spot, x along the pitch length, y along the width.
* one row per frame at `fps` (25 for everything we use).
* missing observations are NaN, for players and ball alike.
* `team[s]` is 0 for the home side and 1 for the away side, per player/track slot.
* `label` uses the scorer's codes: H=0, A=1, D=2, X=3 (X = no ground truth / not graded).
* `period` is 0 for frames outside play (pre-match, halftime, post), else 1 or 2.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")


@dataclass
class Match:
    name: str
    players_xy: np.ndarray          # (T, S, 2) float32, metres
    team: np.ndarray                # (S,) int8, 0 home / 1 away
    ball_xy: np.ndarray             # (T, 2) float32, metres, NaN when missing
    period: np.ndarray              # (T,) int8
    fps: int = 25
    pitch: tuple = (105.0, 68.0)
    players_speed: Optional[np.ndarray] = None   # (T, S) float32, m/s, if provided
    ball_extra: Optional[np.ndarray] = None      # (T, k) any further ball channels, raw
    label: Optional[np.ndarray] = None           # (T,) int8 codes, if known
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.ball_xy)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = {k: v for k, v in self.__dict__.items()
                  if isinstance(v, np.ndarray)}
        np.savez_compressed(path, name=self.name, fps=self.fps, pitch=np.array(self.pitch),
                            meta=np.array(json.dumps(self.meta)), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files if k not in ("name", "fps", "pitch", "meta")}
        return cls(name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(z["pitch"].tolist()),
                   meta=json.loads(str(z["meta"])), **kw)

    def runs(self):
        """[(start, stop)] maximal blocks of one period != 0, half-open.

        Split on every period change, not just on zeros: external data lays the halves out back
        to back, and a decoder must never carry a label across halftime.
        """
        p = np.asarray(self.period)
        change = np.flatnonzero(p[1:] != p[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(p)]))
        return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if p[a] != 0]

    def gt_codes(self):
        """Labels in scorer form: X outside play, so frames between runs are never graded."""
        out = np.where(self.period != 0, self.label, X).astype(np.int8)
        # The official scorer splits runs on X only. Where two periods sit back to back (external
        # data), sacrifice one frame as X so it also sees the halftime boundary.
        for start, _ in self.runs():
            if start > 0 and self.period[start - 1] != 0:
                out[start] = X
        return out


def codes_to_chars(codes):
    return LABEL_CHARS[np.asarray(codes)]


def chars_to_codes(chars):
    lookup = {c: i for i, c in enumerate(LABEL_CHARS)}
    return np.array([lookup[c] for c in chars], dtype=np.int8)


def write_submission(path, codes):
    """frame,label CSV in the exact contract of SUBMISSION.md."""
    chars = codes_to_chars(codes)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{c}\n" for i, c in enumerate(chars))
    return path
