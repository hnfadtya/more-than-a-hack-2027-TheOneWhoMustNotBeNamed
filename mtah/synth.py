"""Synthetic matches with known labels -- for smoke-testing the training loop without any data.

The football here is crude on purpose (formations that drift, a ball that is carried and passed,
spells that end in interceptions or in the ball going out). It only has to be learnable enough to
prove the pipeline end to end: features -> model -> probabilities -> decoder -> official metric.
Never use it to choose anything.
"""

import numpy as np

from .common import A, D, H, Match

HL, HW = 52.5, 34.0


def simulate(name="synth", minutes=20.0, seed=0, fps=25, n_slots=28):
    rng = np.random.default_rng(seed)
    T = int(minutes * 60 * fps)

    # labels: alternating team spells with occasional stoppages
    label = np.empty(T, dtype=np.int8)
    t, team = 0, H
    while t < T:
        if rng.random() < 0.25:
            n = int(rng.uniform(5, 30) * fps)
            label[t:t + n] = D
            t += n
        n = int(max(1.0, rng.exponential(12.0)) * fps)
        label[t:t + n] = team
        t += n
        team = A if team == H else H

    # players: 11 per side around a formation, drifting smoothly; extra slots stay empty (subs)
    team_of = np.array([0] * 11 + [1] * 11 + [0, 1] * ((n_slots - 22) // 2), dtype=np.int8)
    base = np.stack([rng.uniform(-45, -5, 22), rng.uniform(-30, 30, 22)], axis=1)
    base[11:22, 0] *= -1
    # AR(1) drift: a random walk pulled back towards the formation so nobody wanders off forever
    steps = rng.normal(0, 0.08, (T, 22, 2))
    drift = np.empty_like(steps)
    drift[0] = 0
    for k in range(1, T):
        drift[k] = 0.995 * drift[k - 1] + steps[k]
    xy = np.full((T, n_slots, 2), np.nan, dtype=np.float32)
    xy[:, :22] = np.clip(base[None] + drift, [-HL, -HW], [HL, HW])

    # ball: carried by someone of the possessing side, passed every 1-4 s
    ball = np.zeros((T, 2), dtype=np.float32)
    carrier = 0
    t = 0
    while t < T:
        lab = label[t]
        if lab == D:
            n = int(np.argmax(label[t:] != D)) or (T - t)
            side = rng.choice([-1, 1])
            ball[t:t + n] = (rng.uniform(-HL, HL), side * (HW + 1.0))
            t += n
            continue
        side = 0 if lab == H else 1
        mates = np.flatnonzero(team_of[:22] == side)
        if team_of[carrier] != side:
            carrier = int(rng.choice(mates))
        hold = int(rng.uniform(1, 4) * fps)
        stop = min(T, t + hold)
        ball[t:stop] = xy[t:stop, carrier] + rng.normal(0, 0.3, 2)
        t = stop
        if t >= T:
            break
        # next receiver: a mate, or an opponent if the spell ends within the flight
        flight = int(rng.uniform(0.5, 1.5) * fps)
        end = min(T, t + flight)
        nxt = label[min(end, T - 1)]
        pool = np.flatnonzero(team_of[:22] == (1 - side)) if nxt not in (lab, D) else mates
        receiver = int(rng.choice(pool))
        src, dst = ball[t - 1], xy[min(end, T - 1), receiver]
        w = np.linspace(0, 1, end - t, endpoint=False)[:, None]
        ball[t:end] = (1 - w) * src + w * dst
        carrier, t = receiver, end

    period = np.ones(T, dtype=np.int8)
    period[T // 2:] = 2
    return Match(name=name, players_xy=xy, team=team_of, ball_xy=ball, period=period, fps=fps,
                 pitch=(2 * HL, 2 * HW), label=label, meta={"source": "synth", "seed": seed})
