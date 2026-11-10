"""Bring external possession labels closer to the competition's definition of possession.

DFL's BallPossession flips on every touch: in a typical IDSSE match ~29% of team spells last under
2 s. The competition is explicit that those are not changes of possession (CHALLENGE.md):

  * "a simple touch of the ball without control" does not count,
  * "a deflection of a pass that does not intercept it" does not count,
  * "in phases where the ball is contested or loose, possession stays with the last team that had
    it", and after a stoppage "there is always a clear possessor".

`absorb_short_spells` applies exactly that: a team spell shorter than `min_frames` that follows
live play is treated as an uncontrolled touch and handed back to the team that had the ball
before it. A spell that starts from a restart (preceded by D) is kept whatever its length,
because a restart names its possessor.

The threshold is set from that principle (about a second of control), NOT tuned against the
public 5-minute window, which RULES.md forbids.
"""

import numpy as np

from .common import A, D, H


def absorb_short_spells(label, runs, min_frames=25):
    """Copy of `label` with short live-play team spells relabelled to the previous team."""
    out = label.copy()
    if min_frames <= 1:
        return out
    for a, b in runs:
        w = out[a:b]
        if not len(w):
            continue
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(w)]))
        prev_team = None      # team of the last kept spell in live play, None after a stoppage
        for s, e in zip(bounds[:-1], bounds[1:]):
            k = w[s]
            if k == D:
                prev_team = None
                continue
            if k in (H, A) and prev_team is not None and k != prev_team and e - s < min_frames:
                w[s:e] = prev_team          # an uncontrolled touch: possession stays
                continue
            prev_team = w[s]
    return out
