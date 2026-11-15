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

- Colab run3 = randomisation v2 + 60 s windows (`--domain_rand --L 1500 --steps 300`),
  30 epochs x 69 s. Best val 0.7185 (epoch 23, default decoder); tuned decoder plateau ~0.72,
  chose the run2 config min (25, 25, 50), team 8, dead 2 -> 0.7208.
- Stress run2 -> run3: base 0.750 -> 0.727, ball_heavy 0.701 -> 0.699, players_heavy 0.738 -> 0.706,
  false_ball 0.730 -> 0.706, fragment 0.753 -> 0.723, harsh 0.677 -> 0.631,
  **long_gaps 0.678 -> 0.711, atd_like 0.663 -> 0.684**; mean 0.711 -> 0.698.
  -> Rule (atd_like up, base drop < 0.03, mean not down) FAILS on the mean. **run2 stays primary.**
  run3 is a long-gap specialist; it lost on false balls / noisy players (v2 shows those less).
- Pre-registered before running: ensemble run2+run3 (equal weights, run2's decoder, which equals
  run3's) replaces run2 under the same rule: atd_like up, base drop < 0.03, mean over the 8 not down.
  `mtah.stress` now takes "dirA+dirB" specs and caches per-model probabilities.
- Ensemble stress (run2 / run3 rows reproduce the earlier runs exactly), run2 -> run2+run3:
  base 0.750 -> 0.745, ball_heavy 0.701 -> 0.713, players_heavy 0.738 -> 0.735,
  false_ball 0.730 -> 0.733, fragment 0.753 -> 0.746, harsh 0.677 -> 0.664,
  long_gaps 0.678 -> 0.703, **atd_like 0.663 -> 0.697**; mean 0.711 -> **0.717**.
  Rule: atd_like up, base drop 0.006 < 0.03, mean up -> **PASSES. Ensemble run2+run3 is primary.**
- Submission `nn_ens23` (`python -m mtah.make_submission --model outputs/nn/run2 outputs/nn/run3
  --name nn_ens23`, run2's decoder): valid; H 37.4 / A 26.2 / D 36.4%, 306 possessions, mean
  13.2 s, home share 0.59 (close to nn_run2). sha256 `54711f56...d379d1`. Backed up.
- `train.py --rand {off,v1,v2,mix}`. Note: run2 was trained when `--domain_rand` meant v1
  (= `--rand v1` now); run3 = `--rand v2`; `--domain_rand` is kept as an alias of v2.
- Next Colab run4 = `--rand mix --L 1500 --steps 300` (v1 or v2 per sample: a generalist that has
  also seen long ball gaps). Pre-registered: same rule as run3, against the current primary.

### 2026-11-14
- New machine (Apple M1, Python 3.12.16): `nn_ens23` regenerated with `--refresh`, sha256
  `54711f56...d379d1` -- identical to the 11 Nov file. Pipeline reproduces across machines (~1 min).
- Smart Tagging vs our `nn_ens23` predictions (no ground truth involved, aggregates only):
  82% of graded frames covered by a tag; agreement with the tagged team BUILD UP 93%,
  PROGRESSION 85%, transitions/long ball 68-69%, box/final-third/set-piece windows 38-51% (long
  windows that mix dead ball and both teams). "DEFENDING ..." codes mirror the attacking codes
  exactly (same frames, other team). -> Too coarse (median 17-35 s windows) to time hand-overs;
  **Jalur 2 (Smart Tagging as model input) deprioritised.** Kept as a label-free sanity check.
- Jalur 1 started: `mtah/vball.py`. Video 1920x1080, 25 fps, 199319 frames (= submission rows).
  Probe, pretrained COCO YOLO11s at imgsz 1920, 300 frames, vs the ATD ball (2 m radius):
  recall 34-38%, top-1 precision 35-41%, error 0.31 m when right, 3.3 fps on M1 MPS.
  No frame offset. -> Not usable as is; fine-tune on ATD-derived boxes.
- **Pre-registered, before any fine-tuned result:**
  - Detector training data = frames where the ATD sees the ball (automatic labels from the ATD,
    no human input), projected to pixels with the provided homography. Validation = held-out
    time blocks (every 10th minute of video), never trained on. Threshold chosen on those blocks.
  - Stage A (detector): top-1 precision >= 85% within 2 m on the held-out blocks.
  - Stage B (effect): on IDSSE val, `atd_like` with ball gaps refilled at the detector's measured
    rates (fill rate on ATD-missing frames, false rate = 1 - precision, measured error).
    Video ball is used if ensemble spellF1 rises >= 0.01 over plain `atd_like`.
  - Sanity: agreement with Smart Tagging on BUILD UP / PROGRESSION does not drop > 2 points.
  - Known caveat: precision on ATD-visible frames may be optimistic for the gaps (harder frames).
- Dataset (`python -m mtah.vball dataset`): 6000 train / 600 val frames (>= 5 frames apart,
  disjoint minute blocks), 6600 positive + 3219 negative 640x640 tiles, box = 0.6 m at the local
  image scale (median 12 px). Fine-tune notebook `notebooks/colab_vball.ipynb`.
- Stage B response surface (`mtah.stress --vball FILL PREC ERR`, IDSSE val, ensemble run2+run3,
  false detections placed on a random player = worst case): atd_like 0.6970 (reproduces);
  fill 1 / prec 1 **0.7851**; 0.5/0.95 0.7488; 0.5/0.85 0.7271; 0.5/0.7 0.7292; 0.3/0.85 0.7322
  (all err 0.3 m). Differences < ~0.005 are noise. -> Up to +0.09; a mediocre detector still +0.03.
- Visual check (human, 30 tiles): 10 boxes off the ball. Measured why (pretrained YOLO vs ATD,
  aggregates): no frame offset (best k = 0, median 9 px on fast balls); residual median 6 / 9 / 11 px
  for slow / medium / fast balls, ball centre ~2.4 px ABOVE the ATD point when slow (= one ball
  radius at ~21 px/m: the ATD marks the ground point under the ball) and more when bouncing.
  -> `mtah.vball refine`: snap the label to a pretrained detection within 15 px (40.8% of tiles,
  median move 6.2 px), else shift up one ball radius; box 1.0 m (was 0.6). Originals kept in
  labels_atd/.
- Second visual check after refine: ~14/30 judged off (worse). Suspects: snapping onto a
  pretrained false positive (feet/heads next to the ball), a 4 px ball hard to see, or ATD false
  balls. Zoomed crops written (`mtah.vball check --zoom 4` -> outputs/vball/check_zoom/).
- **OPEN (paused 14 Nov evening):**
  1. Human classification of the 30 zoomed crops: A box on ball / B ball near, original ATD dot
     better / C both off but ball near / D no ball at all. B -> drop snapping; C -> label noise,
     train anyway (maybe bigger box); D -> ATD false balls, filter tiles first.
  2. Proposed (NOT yet agreed): Stage A precision 85% -> 70%, in exchange for a harsher Stage B
     (bursty false detections ~12 frames instead of per-frame; precision used = measured - 0.10).
  3. Dataset zip NOT yet uploaded; fine-tune (user, Colab) waits on 1.

### 2026-11-15
- Zoomed crop check (human, 30 tiles after refine): A box on ball 17, B 0, C 0, D no ball 2,
  remaining 11 = ball visible in the crop but far from both the box and the original ATD dot.
  The ATD dot is almost always inside the box (refine moves it only a few px). -> ~37% of
  positive labels are off by more than a box width; this is not a few-pixel bias.
- **Rule revision (agreed with the team, BEFORE any fine-tuned detector exists).** Replaces the
  14 Nov Stage A/B:
  - Stage A: top-1 precision **>= 70%** within 2 m on the held-out minute blocks (was 85%).
    Reason: the IDSSE response surface shows prec 0.7 still gives ~+0.03, so 85% could reject a
    useful detector.
  - Stage B, made harsher to compensate: refill simulated with fill = measured fill rate on
    ATD-missing frames, precision = **measured - 0.10** (gaps are harder frames), false
    detections **bursty (mean 12 frames, one random player per burst)** instead of per-frame,
    err = measured median error. Adopt if ensemble spellF1 >= plain `atd_like` + 0.01.
  - Sanity unchanged: Smart Tagging agreement on BUILD UP / PROGRESSION does not drop > 2 points.
- Harsher Stage B surface (bursty false detections, mean 12 frames on one player), ensemble,
  err 0.3 m: atd_like 0.6970; fill/prec 0.5/0.85 0.7422, 0.5/0.75 0.7360, 0.5/0.6 0.7359,
  0.3/0.6 0.7213, 0.7/0.75 0.7424. Bursty false balls hurt LESS than per-frame ones (0.5/0.85:
  0.742 vs 0.727) -- per-frame flicker makes velocity spikes. Even prec 0.6 gives +0.02-0.04;
  fill rate matters more than precision in this range.
- Recount on the zoomed crops (the earlier 17/0/0/2 was read off the unzoomed folder):
  A 18, B 0, C 0, D 3, rest off. Split by refine outcome: **snapped (ATD + pretrained detector
  agree) 15/15 correct; unsnapped (ATD alone) 1/15 correct, 2 marginal, 12 off.** -> The ATD
  ball is itself wrong on a large share of the frames where it reports one (likely predicted /
  interpolated). Labels are trustworthy only where the ATD and a detector agree.
- Consequence for Stage A (noted, rule NOT changed): precision is measured against the ATD ball,
  which is often wrong, so measured precision UNDERESTIMATES the detector; the rule is
  conservative.
- Dataset v2: 15000 / 1500 frames sampled, `refine --drop_unsnapped` keeps only snapped
  positives and drops the negative tile of every dropped frame (the real ball may be in it).
  Plan: ft1 on v2 -> use ft1 to re-snap more frames (harder balls) -> ft2 (self-training,
  labels still 100% algorithmic).
- v2 built: train 6122 pos / 3101 neg, val 585 pos / 288 neg (snapped 40.6%). Human check of
  check_v2 OK. ft1 = YOLO11s fine-tuned on v2 (Colab, user), best.pt sha256 `a07e9c5c...5698f`.
- **Pre-registered before evaluating ft1:** Stage A threshold = the LOWEST tau in
  {0.05, 0.1, 0.2, 0.3, 0.5, 0.7} whose top-1 precision (2 m, held-out minute blocks, imgsz 1920)
  is >= 70% (fill rate matters more than precision in the Stage B surface). If none passes, ft1
  fails Stage A. Probe: 40 segments x 15 frames, seed 0, val blocks only.
- **Stage A ft1: PASS.** 600 val frames, imgsz 1920. tau 0.05: top-1 precision 74.4% (pretrained
  35%), recall 68.4% (38%), err median 0.62 m vs the (noisy) ATD, detection on 53.9% of
  ATD-missing frames. Precision flat over tau (74.4-76.0%), fill falls 53.9% -> 25.7%. No frame
  offset. 7.0 fps on M1 MPS. -> tau = 0.05 per the rule.
- **Stage B ft1: PASS.** fill 0.539, prec 0.744 - 0.10 = 0.644, err 0.62, bursty false (12):
  ensemble spellF1 atd_like 0.6970 -> **0.7313 (+0.034 >= 0.01)**.
- Remaining before adoption: full-match fill on BAR-ATH (top-1 per ATD-missing graded frame,
  tau 0.05, no extra filtering -- exactly what was measured) and the Smart Tagging sanity check.
- `python -m mtah.vball fill --tag ft1` -> `cache/BAR-ATH_vball_ft1.npz` (64643 gap frames,
  positions rounded to 1 cm, conf to 1e-3, checkpoint/resume). Speed on M1: MPS 7.7 fps
  (~2.3 h), CPU 1.8 fps (~10 h). Smoke test 48 frames: MPS vs CPU **identical** after rounding.
  `make_submission --vball ft1` fills only frames where the ATD has no ball.
- Full fill done (2026-11-15 01:18 WIB, ~2.7 h on MPS, slowed 7.2 -> 6.5 fps with heat):
  35023 detections on 64643 gap frames (54.2%, matches Stage A 53.9%).
  `cache/BAR-ATH_vball_ft1.npz` sha256 `5b5ca25e41654b465d2bbd4994113bf9e98a9012aa489cadbd98fc8f05fb333f`
  (frame-level derived data: never committed; backed up on Drive).

### 2026-11-15 - video ball adopted: nn_ens23_vb1
- `python -m mtah.make_submission --model outputs/nn/run2 outputs/nn/run3 --vball ft1 --name nn_ens23_vb1`
  (61 s on CPU). Ball seen on graded frames 59.3% -> 81.4%. Graded: H 36.7 / A 26.3 / D 37.0%,
  304 possessions, mean 13.2 s, home share 0.58 (nn_ens23: 37.4 / 26.2 / 36.4%, 306). 4.9% of
  graded frames change label.
- **Sanity check (pre-registered: BUILD UP / PROGRESSION agreement must not drop > 2 points): PASS.**
  BUILD UP 93.1% -> 92.9% (-0.2), PROGRESSION 84.9% -> 85.0% (+0.2). Other codes move by < 1 point
  in "pred = tagged team" except SET PIECES -1.1 and GOALS -2.6, where the shift is to D, not to the
  other team (goal celebrations / restarts are dead ball). "pred = other team" falls on the attacking
  codes (BALL IN THE BOX 29.3 -> 27.8%, CREATING CHANCES 23.5 -> 21.5%).
- -> **Primary submission: `nn_ens23_vb1.csv`**, sha256
  `ee60898ca2a090b97b01fe990f68af7cec552cdb1f97b94c6593d726a45fce60`. Backup in `submissions_backup/`.
  Previous primary nn_ens23 (`54711f56...`) kept as fallback.
- Reproduction chain: `vball fill --tag ft1` (ft1 best.pt `a07e9c5c...5698f`, needs the tactical video)
  -> `make_submission ... --vball ft1`.

### 2026-11-15 - post-decoder study (written BEFORE any result is seen)
Context: the repo was reproduced on a new machine (all hashes match). Two cheap candidates that
need no retraining, judged on IDSSE validation only (J03WMX + J03WR9, `mtah.stress` fixed seeds,
the 8 scenarios), model = ensemble run2+run3. Baseline = current decoder. Ideas come from the
public repos of other teams (DSBA: touch snapping; YT / 2hClub: D share), re-implemented, not copied.
- Study 1, dead-ball share. Diagnostic first: predicted vs ground-truth H/A/D share on graded
  frames and number of D runs, per scenario. Candidate: additive bias b on log p(D) before the
  decoder, b in {-0.3, -0.15, +0.15, +0.3}. A share gap alone is NOT evidence to move: IDSSE's D is
  DFL's BallStatus and the competition's D share is unknown.
- Study 2, touch snapping (post-decoder). Touch = frame where |v(t+3) - v(t-3)| >= 4 m/s (ball
  velocity from 1-frame differences) with a player within 2.0 m (values documented by DSBA, not
  tuned). Each team switch (H<->A) and each restart (D -> H/A) boundary moves to the first touch by a
  player of the NEW team within +-W frames, if both neighbours keep their min duration.
  W in {12, 25, 50} frames. Evaluated on the ball track as the model sees it (corrupted).
  Caveat: IDSSE labels flip at DFL touches; the competition's convention is not known.
- Adoption rule (both studies, same as above): mean spellF1 over the 8 scenarios up by >= 0.005
  AND atd_like up AND base not down by more than 0.01. Among passing options take the best mean,
  ties -> the smaller |b| / W. Study 2 is run on top of Study 1's outcome (b = 0 if not adopted).
  If nothing passes, the current decoder stays. No other options, no re-tuning after the result.
- The public 5 minutes are printed once, after the decision, as a pipeline check only.

- Result (`python -m mtah.post_study`, outputs/post_study.json). Baseline reproduces the earlier
  ensemble stress exactly (mean 0.7170, atd_like 0.6970). Study 1a: predicted vs ground-truth shares
  on graded frames agree in every scenario (D 0.417-0.420 vs 0.417 gt; H 0.31, A 0.27), so the model
  is not biased on D on IDSSE even under atd_like. Study 1b: D bias b = -0.30 / -0.15 / +0.15 /
  +0.30 -> mean 0.7172 / 0.7168 / 0.7159 / 0.7169, atd_like 0.6959 / 0.6941 / 0.6933 / 0.6939: none
  passes (atd_like down, gains < 0.005). Study 2: touch snapping W = 12 / 25 / 50 -> mean 0.7146 /
  0.7068 / 0.6830, atd_like 0.6932 / 0.6864 / 0.6582, base 0.7450 / 0.7381 / 0.7162: all worse, the
  loss grows with W. **Neither is adopted; the decoder stays as is and nn_ens23 is unchanged.**
  Not measured: the precision of the touch detector itself; the competition GT's switch convention
  (touch vs pass) is still unknown. The D share of BAR-ATH (36.4%) vs IDSSE gt (41.7%) is a
  property of the two matches / label sources, not evidence of a model bias.

### 2026-11-15 - ft2 self-training (written BEFORE any result is seen)
Idea: ft1 finds balls the COCO detector misses, so re-snapping ATD labels with ft1 keeps more
(and harder) trustworthy labels than v2 (40.6% snapped). Labels stay 100% algorithmic: a label is
kept only where the ATD ball and ft1 agree within 15 px.
- Dataset v3: fresh sample, `dataset --out outputs/vball/dataset_v3 --n_train 24000 --n_val 2400
  --gap 3 --seed 1`, then `refine --weights outputs/vball/ft1/best.pt --drop_unsnapped` (conf 0.03,
  snap 15 px, box 1.0 m, all as v2). Train/val from the same disjoint minute blocks (`is_val`).
- Label check before training: 30 random positive tiles of v3 (`check --zoom 4`, seed 0), human
  eyeball. Continue only if >= 27/30 have the box on the ball; otherwise ft2 is dropped.
- ft2 = YOLO11s from the COCO weights (not from ft1, to limit confirmation bias), same notebook
  settings as ft1 except epochs 40 (2.5x more data): imgsz 640, batch 64, seed 0, deterministic.
- Stage A: identical to ft1 (probe on val blocks, 40 segments x 15 frames, seed 0, imgsz 1920,
  radius 2 m). tau = the lowest in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7} with top-1 precision >= 70%.
- Stage B: identical spec (fill = measured ATD-missing detection rate, prec = measured - 0.10,
  err = measured, bursty false balls of 12). **ft2 replaces ft1 only if its Stage B spellF1 >=
  0.7313 + 0.005 = 0.7363.** Otherwise ft1 / nn_ens23_vb1 stays.
- If adopted: full fill `--tag ft2` (on Colab GPU only if a 48-frame smoke test is identical to
  the M1 output after rounding, else on MPS), `make_submission ... --vball ft2 --name nn_ens23_vb2`,
  then the same Smart Tagging sanity check vs nn_ens23 (BUILD UP / PROGRESSION not down > 2 points).
- Cut-off: anything not passed by 16 Nov 23:00 WIB is not used. No other variants of ft2.
- v3 built (19 min on M1): 26400 sampled frames, snapped to ft1 48.5% (v2 with COCO: 40.6%),
  snap |shift| median 6.3 px. train 11688 pos / 5779 neg, val 1110 pos / 546 neg (v2: 6122 / 585).
- **Label check: 30/30 boxes on the ball (>= 27 required) -> continue to training.**
- ft2 trained on Colab (user), best.pt sha256 `ce0d1b40b2c865064bec8011e5ace155c01dd6e3cc78025ba63b5805aae9d777`.
- **Stage A ft2: PASS but no better than ft1.** tau 0.05: top-1 precision 74.9% (ft1 74.4%),
  recall 70.5% (68.4%), err 0.63 m (0.62), ATD-missing detection 53.5% (53.9%). Precision flat
  over tau again (74.8-75.2%). Doubling the labels did not move the detector: the ceiling is
  likely the ball itself (tiny, occluded, motion blur at 1920 px) or the ATD reference, not data.
- **Stage B ft2: FAIL.** vball_f0.535_p0.649_e0.63_b12 -> 0.7225 < 0.7363 required (ft1: 0.7313).
  -> **ft2 not adopted; ft1 / nn_ens23_vb1 stays.** Note: ft1 and ft2 have near-identical inputs
  to the simulation yet differ by 0.009, so Stage B has ~0.01 simulation noise between nearby
  settings (different random draws). ft1's margin over atd_like (+0.034) is well above it; a
  ft1-vs-ft2 margin of 0.005 would not have been.
- **Colab GPU fill is NOT reproducible against the M1** (ft1 weights, first 2000 gap frames,
  Colab CUDA batch 32 vs M1 MPS batch 8): detections 1000 vs 981 (19 only on Colab, 0 only on M1),
  xy identical on 61% of common frames (max 17 m), conf identical on 14% (max |d| 0.25). MPS vs
  CPU was identical (48 frames). -> per the rule no Colab fill is used. For the repo: the fill must
  be regenerated on CPU or Apple MPS; CUDA gives a different (similar-quality) ball file. Cause not
  isolated (CUDA kernels / TF32, batch size, or video decode build).

## Next
- [ ] When data lands: `python -m mtah.atd` stats -> calibrate CorruptCfg; first submissions.
- [ ] Smart Tagging as extra evidence (BAR-ATH only; set pieces -> restarts / dead ball).
- [ ] Measure ATD noise statistics when data arrives -> set `CorruptCfg`.
- [ ] Heuristic evidence v1 (dead ball, sustained proximity, Smart Tagging) + decoder grid search
      on IDSSE val probs.
