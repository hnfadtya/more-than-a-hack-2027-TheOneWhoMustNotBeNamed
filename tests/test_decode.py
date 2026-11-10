"""Decoder sanity checks on synthetic sequences -- no match data needed.

    python -m pytest tests -q        (or)        python tests/test_decode.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.common import A, D, H, X, Match  # noqa: E402
from mtah.decode import decode_runs, viterbi, votes_to_logp  # noqa: E402
from mtah.scoring import score  # noqa: E402


def segments_truth(rng, n_seg=60):
    """A plausible possession sequence: H/A spells of 2-40 s with D stoppages between some."""
    out, team = [], H
    for _ in range(n_seg):
        if rng.random() < 0.25:
            out += [D] * int(rng.integers(5 * 25, 30 * 25))
        out += [team] * int(rng.integers(2 * 25, 40 * 25))
        team = A if team == H else H
    return np.array(out, dtype=np.int8)


def noisy_votes(truth, rng, flip=0.25, missing=0.41, burst=40):
    """Per-frame votes with random flips and bursty missingness (like the ATD ball)."""
    votes = truth.copy()
    flips = rng.random(len(truth)) < flip
    votes[flips] = rng.integers(0, 3, flips.sum())
    gone = np.zeros(len(truth), bool)
    t = 0
    while t < len(truth):
        if rng.random() < missing / burst:
            gone[t:t + int(rng.integers(1, 2 * burst))] = True
        t += 1
    votes[gone] = X
    return votes


def test_clean_input_is_returned_unchanged():
    truth = np.array([H] * 100 + [D] * 50 + [A] * 80, dtype=np.int8)
    out = viterbi(votes_to_logp(truth), min_dur=(25, 25, 25), switch=2.0)
    assert np.array_equal(out, truth)


def test_min_duration_is_respected():
    rng = np.random.default_rng(0)
    truth = segments_truth(rng, 20)
    out = viterbi(votes_to_logp(noisy_votes(truth, rng)), min_dur=(50, 50, 50), switch=1.0)
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([-1], out, [-1]))) != 0))
    assert runs[1:-1].min() >= 50  # interior runs; the edges may be cut by the block


def test_no_evidence_is_carried_by_context():
    truth = np.array([H] * 200 + [X] * 300 + [H] * 200, dtype=np.int8)
    out = viterbi(votes_to_logp(truth), min_dur=(10, 10, 10), switch=3.0)
    assert (out == H).all()


def test_runs_never_leak_across_halftime():
    period = np.array([0] * 10 + [1] * 100 + [2] * 100, dtype=np.int8)
    m = Match("t", np.zeros((210, 1, 2), np.float32), np.zeros(1, np.int8),
              np.zeros((210, 2), np.float32), period)
    assert m.runs() == [(10, 110), (110, 210)]
    votes = np.array([X] * 10 + [H] * 100 + [A] * 100, dtype=np.int8)
    out = decode_runs(votes_to_logp(votes), m.runs(), (30, 30, 30), 5.0)
    assert (out[:10] == X).all() and (out[10:110] == H).all() and (out[110:] == A).all()


def test_decoding_beats_raw_votes_on_spellf1():
    rng = np.random.default_rng(1)
    truth = segments_truth(rng, 80)
    votes = noisy_votes(truth, rng)
    raw = np.where(votes == X, D, votes).astype(np.int8)
    dec = viterbi(votes_to_logp(votes), min_dur=(25, 25, 50), switch=4.0)
    s_raw, s_dec = score(truth, raw), score(truth, dec)
    print(f"raw spellF1 {s_raw['spell_f1_mean']:.3f} -> decoded {s_dec['spell_f1_mean']:.3f}")
    assert s_dec["spell_f1_mean"] > s_raw["spell_f1_mean"] + 0.3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
