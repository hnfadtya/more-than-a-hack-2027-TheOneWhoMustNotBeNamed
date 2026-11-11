# Worklog — More than a Hack 2027

Deadline: **17 Nov 2026, 23:59 CET** (= 18 Nov 05:59 WIB). Submission instructions arrive by 14 Nov.
Competition data expected by ~12 Nov.

## Plan
- **A. Heuristics + decoder (CPU).** Dead-ball detection, sustained ball proximity, gap filling,
  then `mtah.decode` (Viterbi, min durations, switch costs). First submission that beats 0.2433.
- **B. Learned model (Colab A100).** Trained on IDSSE (other games) with ATD-like corruption
  (bursty ball dropout, broken tracks, shuffled slots). Per-frame H/A/D probs -> same decoder.
- **C. Optional.** Our own ball detection from the tactical video to fill the 41% missing ball.
- Validation = IDSSE matches (corrupted), scored with the official metric code
  (`mtah.scoring`). The public 5 minutes are a pipeline check only — never tuned on.

## Hard rules to keep in mind
- No hand labels / corrections on BAR-ATH. No tuning on the public 5 min.
- No competition data (or extracts) to any hosted LLM.
- Repo must regenerate the exact submission file: final command = deterministic inference
  from saved weights + decoder.

## Status
### 2026-11-09
- `.venv` (Python 3.12) with pinned `requirements-mtah.txt`.
- `mtah/common.py` shared `Match` format; `mtah/atd.py` ATD parser + cache (untested — no data).
- `mtah/decode.py` Viterbi H/A/D with exact min durations, numba, ~0.08 s per half.
  `tests/test_decode.py` passes (synthetic: spellF1 0.00 raw -> 0.95 decoded).
- `mtah/scoring.py` official metrics on any arrays.
- `mtah/external/idsse.py` IDSSE -> Match with H/A/D labels (untested — needs manual download,
  see `external/README.md`).

- Training scaffold (Jalur B), smoke-tested end to end on synthetic matches (CPU):
  - `mtah/augment.py` ATD-like corruption (bursty ball/player dropout, jitter, flips, team swap).
    Defaults are guesses until ATD stats are measured.
  - `mtah/features.py` 24 nearest players (set, no IDs) + ball token per frame.
  - `mtah/model.py` PossessionNet: per-frame set transformer -> dilated TCN -> BiGRU -> H/A/D.
  - `mtah/train.py` leave-match-out validation scored as the board does (probs -> Viterbi ->
    official spellF1); best.pt by val spellF1; resumes from last.pt automatically.
  - `mtah/infer.py` CPU inference for BAR-ATH (competition data stays local), probs rounded
    before decoding for reproducibility.
  - `notebooks/colab_train.ipynb` clone (GH_TOKEN secret) -> train -> outputs on Drive.

### 2026-11-10
- All 7 IDSSE matches downloaded and converted (`cache/idsse_*.npz`, ~1M labelled frames).
- Finding: DFL possession flips on every touch (median spell 3.8 s, 29% < 2 s). The challenge
  says uncontrolled touches/deflections are not changes of possession -> `mtah/labels.py`
  absorbs live-play spells < `min_spell` (default 25 frames = 1 s, chosen from the challenge
  text, NOT from the public window) into the previous team. Applied to train and val labels.
- Decision: Jalur A is the primary path, Jalur B (Colab) the upside. Pick by IDSSE validation.
- Validation = IDSSE J03WMX + J03WR9, corrupted once with a fixed seed.
- `mtah/reference.py`: official baseline algorithm on that validation: smooth0 0.013,
  smooth25 0.131, **smooth50 0.208**, smooth125 0.198 — same ranking and similar level as on the
  real match (0.049 / 0.200 / 0.243 / 0.224), a first sign the corruption model is realistic.
- Jalur A = `mtah/handfeat.py` (84 football features) + `mtah/gbdt.py` (HistGradientBoosting):
  **val spellF1 0.4615** with the default decoder (~2.2x the baseline on the same data).
  Most used features: closest-player distance, ball speed, overall player speed (dead ball).
- `mtah/tune_decoder.py`: decoder grid search on validation probs (any model).
- Jalur B training run started on Colab (`run1`).

- Decoder grid (460 configs) on GBDT v1 val probs: flat plateau at min team spell 50-75 frames,
  team switch cost 8-12 (0.523-0.528); dead-ball switch cost irrelevant. DEFAULT_DECODE set to
  the plateau middle: min_dur (50, 50, 100), switch team 8 / dead 2 -> **val spellF1 0.5244**
  (vs 0.4615 with the old default, baseline 0.208). Slightly optimistic: tuned on the same two
  validation matches.

- Jalur B `run1` (Colab A100, 30 epochs x 189 s, 1.08M params): best val spellF1 **0.7485**
  (epoch 16), plateau ~0.74 from epoch ~10, no overfitting. Weights in `outputs/nn/run1/`.
- Decoder tuned per model (`decoder.json` next to each model):
  NN 0.7510 (min 25 frames, team switch 4 — the net is already smooth),
  GBDT 0.5244 (min 50/100, team switch 8). Ensembles: 50/50 0.691, 75/25 0.746 -> no gain on IDSSE.
- `mtah/make_submission.py`: ATD -> probs (cached) -> decoder -> CSV -> official validation ->
  possession stats + public-window pipeline check. Tested end to end on a FAKE ATD built from a
  synthetic match (scratchpad, not in data/): parse 9 s, GBDT 7 s, NN 154 s on CPU, both files valid.

- Stress test (`mtah/stress.py`, IDSSE val, unseen corruptions; spellF1 NN / GBDT):
  base 0.748 / 0.516, ball_heavy 0.685 / 0.479, players_heavy 0.585 / 0.357,
  false_ball 0.597 / 0.472, fragment 0.740 / 0.484, harsh 0.420 / 0.241.
  -> NN wins every scenario; robust to track fragmentation; weakest on false balls (over-splits
  possessions) and noisy/missing players. **NN is the primary model.**
- Base corruption reproduces the ATD's published stats almost exactly (20.5 objects/frame,
  ball seen 56.6% vs 20.5 / 56.9% in DATA.md).
- Added to `mtah/augment.py`: false ball, track fragmentation (slot reuse, 3% wrong team, 3%
  duplicates), and `randomized()` domain randomisation; `train.py --domain_rand`.

### 2026-11-11
- Moved to a standalone private repo (origin); old fork kept as `fork`, Metrica as `upstream`.
- Colab run2 = run1 + `--domain_rand` (71 s/epoch this time). Val spellF1 (tuned decoder)
  0.7377 vs run1 0.7510 -> base drop 0.013 (< 0.03 allowed).
- Stress test (pre-registered rule: run2 wins if mean over the 6 scenarios is higher and base drops
  < 0.03), spellF1 run1 -> run2:
  base 0.748 -> 0.750, ball_heavy 0.685 -> 0.701, players_heavy 0.585 -> 0.738,
  false_ball 0.597 -> 0.730, fragment 0.740 -> 0.753, harsh 0.420 -> **0.677**;
  mean 0.629 -> **0.725**. **run2 is the primary model.** Decoder: min (25, 25, 50), team 8, dead 2.
- `mtah/noise_stats.py`: ATD vs simulated corruption, aggregate statistics only.

- Competition data unpacked (non-video). `mtah.atd` reproduces DATA.md exactly: 199319 frames,
  46 slots, ball seen 56.9% all / 59.3% graded, 20.5 objects per graded frame.
- `mtah.noise_stats` (aggregates only), ATD vs our corruption: the missing FRACTION matched but
  the STRUCTURE did not — ATD ball gaps median/mean/p90 268/431/845 frames (11/17/34 s) vs ours
  29/40/91; visible runs 345/632 vs 34/50; ball rarely seen outside the pitch (2.7% vs 9.8%);
  nearest player 1.9 m vs 2.7 m; almost no false balls; 9.2% of frames with < 18 players.
- First submissions (both pass the official validator):
  - `gbdt_v1`: collapses on ATD (93% D, 123 possessions) -> **Jalur A dropped** (judged from the
    implausible output distribution, not from any ground truth).
  - `nn_run2`: H 37.5 / A 25.8 / D 36.7%, 307 possessions, mean 13.1 s, home share 0.59.
    Public-window pipeline check spellF1 0.73 (baselines there 0.33-0.42) — a check, not a target.
    Backed up in `submissions_backup/` (gitignored).
- New stress scenarios matched to the measured ATD: `long_gaps` (ball_burst 430) and `atd_like`
  (+ ball lost when far from players / outside the pitch, end of pitch out of view, fragmentation).
  run2: base 0.750 -> long_gaps 0.678, atd_like 0.663 (~-11%).
- `augment.randomized` v2 (v1 kept as `randomized_v1`): 70% long-gap regime (150-900 frames),
  far/out-of-pitch boosts, field-of-view dropout, fewer false balls, smoother ball.
  `markov_missing` compiled with numba (bit-identical output, verified).

## Next
- [ ] Colab run3: `--domain_rand --L 1500 --steps 300` (randomisation v2, 60 s windows).
  Pre-registered rule: run3 replaces run2 if atd_like improves, base drops < 0.03 and the mean
  over all 8 stress scenarios does not drop.
- [ ] When data lands: `python -m mtah.atd` stats -> calibrate CorruptCfg; first submissions.
- [ ] Smart Tagging as extra evidence (BAR-ATH only; set pieces -> restarts / dead ball).
- [ ] Measure ATD noise statistics when data arrives -> set `CorruptCfg`.
- [ ] Heuristic evidence v1 (dead ball, sustained proximity, Smart Tagging) + decoder grid search
      on IDSSE val probs.
