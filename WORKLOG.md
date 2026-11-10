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

## Next
- [ ] Download IDSSE (browser), run `python -m mtah.external.idsse`, sanity-check labels.
- [ ] First real Colab run on IDSSE.
- [ ] Measure ATD noise statistics when data arrives -> set `CorruptCfg`.
- [ ] Heuristic evidence v1 (dead ball, sustained proximity, Smart Tagging) + decoder grid search
      on IDSSE val probs.
