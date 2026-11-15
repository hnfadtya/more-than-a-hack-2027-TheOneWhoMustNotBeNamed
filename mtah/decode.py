"""Turn per-frame evidence into whole possessions: a Viterbi decoder over H / A / D.

Why this exists: spellF1 scores possessions as intervals, so a per-frame classifier that flickers
H H A H H loses a whole possession even when its frame accuracy is fine. The decoder finds the
single best label sequence under three constraints:

  * evidence      -- `logp[t, k]`, a log-score for label k at frame t (H=0, A=1, D=2). A frame with
                     no information should have equal scores (e.g. all 0) and is then decided
                     entirely by its neighbours.
  * switch costs  -- `switch[a, b]` is subtracted every time the label changes a -> b. Asymmetric
                     on purpose: H->A (a turnover) and A->D (ball goes out) are different events.
  * min durations -- `min_dur[k]` frames: once label k starts it must last at least that long.
                     Implemented exactly, by expanding each label into a chain of counter states
                     (an explicit-duration HMM truncated at min_dur), so no smoothing heuristic
                     is involved and the result is the true optimum.

Cost: O(T * sum(min_dur)) time, compiled with numba; a full half decodes in well under a second.
Deterministic -- ties are broken by the lowest state index -- which the reproducibility rule needs.
"""

import numpy as np
from numba import njit

from .common import X

N_LAB = 3
NEG_INF = -1e18


@njit(cache=True)
def _viterbi(logp, min_dur, switch):
    T = logp.shape[0]
    off = np.zeros(N_LAB + 1, dtype=np.int64)
    for k in range(N_LAB):
        off[k + 1] = off[k] + min_dur[k]
    S = off[N_LAB]
    lab = np.empty(S, dtype=np.int64)
    for k in range(N_LAB):
        for c in range(min_dur[k]):
            lab[off[k] + c] = k

    score = np.full(S, NEG_INF)
    new = np.empty(S)
    back = np.empty((T, S), dtype=np.int16)

    # t = 0: a segment may have started before the run did, so its first frame can be either
    # fresh (counter 0) or already mature (counter K-1).
    for k in range(N_LAB):
        score[off[k]] = logp[0, k]
        score[off[k] + min_dur[k] - 1] = logp[0, k]
    for s in range(S):
        back[0, s] = s

    for t in range(1, T):
        for m in range(N_LAB):
            K = min_dur[m]
            first = off[m]
            mature = off[m] + K - 1
            # entry state (m, 0): from the mature state of another label, paying the switch cost
            best = NEG_INF
            arg = first
            for k in range(N_LAB):
                if k == m:
                    continue
                v = score[off[k] + min_dur[k] - 1] - switch[k, m]
                if v > best:
                    best = v
                    arg = off[k] + min_dur[k] - 1
            if K == 1:  # entry state is also the mature state, so it may stay
                if score[first] > best:
                    best = score[first]
                    arg = first
            new[first] = best + logp[t, m]
            back[t, first] = arg
            # counting-up states
            for c in range(1, K):
                s = first + c
                best = score[s - 1]
                arg = s - 1
                if s == mature and score[s] > best:
                    best = score[s]
                    arg = s
                new[s] = best + logp[t, m]
                back[t, s] = arg
        for s in range(S):
            score[s] = new[s]

    s = 0
    for j in range(1, S):
        if score[j] > score[s]:
            s = j
    out = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        out[t] = lab[s]
        s = back[t, s]
    return out


def _as_switch(switch):
    if np.isscalar(switch):
        m = np.full((N_LAB, N_LAB), float(switch))
    else:
        m = np.asarray(switch, dtype=np.float64).reshape(N_LAB, N_LAB).copy()
    np.fill_diagonal(m, 0.0)
    return m


def viterbi(logp, min_dur=(1, 1, 1), switch=0.0):
    """Best H/A/D sequence for one contiguous block of frames. Returns int8 codes (T,)."""
    logp = np.ascontiguousarray(np.nan_to_num(np.asarray(logp, dtype=np.float64), nan=0.0))
    if logp.ndim != 2 or logp.shape[1] != N_LAB:
        raise ValueError(f"logp must be (T, 3), got {logp.shape}")
    min_dur = np.maximum(1, np.asarray(min_dur, dtype=np.int64))
    if min_dur.sum() >= np.iinfo(np.int16).max:
        raise ValueError("sum(min_dur) too large")
    if len(logp) == 0:
        return np.empty(0, dtype=np.int8)
    return _viterbi(logp, min_dur, _as_switch(switch))


def decode_runs(logp, runs, min_dur=(1, 1, 1), switch=0.0):
    """Decode each [start, stop) run independently (halves never leak into each other).

    Frames outside every run come back as X, which is free: they are never graded.
    """
    out = np.full(len(logp), X, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = viterbi(logp[start:stop], min_dur, switch)
    return out


def votes_to_logp(codes, n_classes=N_LAB, eps=0.05):
    """Hard per-frame codes (with anything >= n_classes meaning 'no evidence') -> log-scores.

    A frame voting k gets log(1-eps) for k and log(eps/2) for the others; a frame with no vote gets
    zeros, i.e. no opinion. Handy for decoding a heuristic's raw output.
    """
    codes = np.asarray(codes)
    logp = np.zeros((len(codes), n_classes))
    known = codes < n_classes
    logp[known] = np.log(eps / (n_classes - 1))
    logp[np.flatnonzero(known), codes[known].astype(np.int64)] = np.log(1 - eps)
    return logp


PROB_DECIMALS = 4
# Chosen by mtah.tune_decoder on IDSSE validation (GBDT v1 probs), from the middle of a flat plateau
# (min team spell 50-75 frames, team switch cost 8-12 all within 0.005 spellF1), not its argmax.
# Team switches are expensive (a turnover needs sustained evidence); dead-ball switches are cheap.
DEFAULT_DECODE = dict(min_dur=(50, 50, 100),
                      switch=((0.0, 8.0, 2.0), (8.0, 0.0, 2.0), (2.0, 2.0, 0.0)))


def probs_to_codes(probs, runs, min_dur, switch, d_bias=0.0):
    """Per-frame H/A/D probabilities (T,3) -> decoded codes, X outside runs.

    Rounded first, so last-bit float differences between machines cannot flip a decision; NaN rows
    (no prediction) become uniform, i.e. no opinion. `d_bias` is added to log p(D) (0 = off).
    """
    p = np.round(np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=1 / 3), PROB_DECIMALS)
    logp = np.log(np.clip(p, 1e-4, 1.0))
    if d_bias:
        logp[:, 2] += d_bias
    return decode_runs(logp, runs, min_dur, switch)
