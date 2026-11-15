# CLAUDE.md — handoff for Claude Code

Read this first, then `WORKLOG.md` (daily log, all results, pre-registered rules) and `CHALLENGE.md`.
The user writes in Indonesian; answer in Indonesian. Code, comments and docs stay in English.

## Competition
More than a Hack 2027 — Phase 1: Ball Possession (Metrica Sports / FC Barcelona / MWC).
For each of the 199,319 frames of BAR-ATH (Barcelona = H, Athletic = A) output H/A/D/X.
Ranked on spellF1 (possession spells matched by temporal IoU 0.3-0.7), tie-break macroF1.
Baseline floor 0.2433. **Deadline 17 Nov 2026 23:59 CET (= 18 Nov 05:59 WIB).** The Google Form
link arrives by email by 14 Nov. Deliverables: `<team_name>.csv`, a public repo that regenerates
that exact file, and a technical doc `<team_name>.pdf` or `.md`. Organisers:
hackathon.talentarena@mobileworldcapital.com (GitHub issues on the Metrica repo are not an option).

## Hard rules — never break these
- **No competition data, or any extract of it, to any hosted LLM (that includes you).** Code that
  touches `data/` runs on the user's machine; only scores and aggregate statistics come back.
  Never print raw values, sample rows or frames of BAR-ATH. No Gemini in Colab notebooks that hold
  competition data.
- No hand labels / corrections on BAR-ATH, no human-produced data about the game.
- **Never tune or select on the public 5-minute window.** It is printed as a pipeline check only.
  Models and decoder settings are chosen on IDSSE validation with rules written in WORKLOG.md
  *before* the result is seen.
- **Never `git commit` or `git push`.** The user commits. Never put tokens (GH_TOKEN) in chat,
  notebooks or files.
- `data/` (competition data) is never committed or redistributed.

## Approach (details and numbers in WORKLOG.md)
- External training data: IDSSE open data (Bassek et al., CC BY 4.0), 7 DFL matches with
  possession labels -> `mtah/external/idsse.py` -> `cache/idsse_<id>.npz`. DFL possession flips on
  every touch, so live-play spells < 25 frames are absorbed (`mtah/labels.py`, from the challenge
  text, not tuned).
- Train on IDSSE corrupted to look like the ATD (`mtah/augment.py`: bursty ball/player dropout,
  long ball gaps, field-of-view loss, false balls, track fragmentation, domain randomisation).
- `PossessionNet` (`mtah/model.py`, 1.08M params) -> per-frame H/A/D probs -> Viterbi decoder with
  min durations and switch costs (`mtah/decode.py`), per-model `decoder.json`.
- Validation: IDSSE J03WMX + J03WR9, fixed-seed corruption, official metric (`mtah/scoring.py`).
  Robustness: `mtah/stress.py`, 8 unseen-corruption scenarios (`atd_like` matches the measured ATD
  noise from `mtah/noise_stats.py`).
- Training on the user's Colab A100 via `notebooks/colab_train.ipynb` (outputs on Google Drive
  `MyDrive/Metrica-Hackathon/mtah/outputs/<run>/`). Inference on BAR-ATH: local CPU, deterministic.

## Current state (2026-11-15)
- **Primary: `nn_ens23_vb1`** = the ensemble below + video ball (YOLO11s ft1 fills ATD ball gaps,
  `mtah/vball.py`; Stage A/B + Smart Tagging sanity PASS, WORKLOG 2026-11-14/15).
  `python -m mtah.vball fill --tag ft1` (needs tactical video, ~2.7 h MPS) then
  `python -m mtah.make_submission --model outputs/nn/run2 outputs/nn/run3 --vball ft1 --name nn_ens23_vb1`
  -> sha256 `ee60898ca2a090b97b01fe990f68af7cec552cdb1f97b94c6593d726a45fce60`
  (H 36.7 / A 26.3 / D 37.0%, 304 possessions). ft1 weights `outputs/vball/ft1/best.pt`
  sha256 `a07e9c5c1c53996877dd5d2ab76e735bc2e5c643c97c3a1020528c813bd5698f`.
- Fallback: ensemble run2+run3 (equal weights, run2's decoder: min (25,25,50), team switch 8,
  dead 2). Passed the pre-registered rule (stress mean 0.711 -> 0.717, atd_like 0.663 -> 0.697).
  `python -m mtah.make_submission --model outputs/nn/run2 outputs/nn/run3 --name nn_ens23`
  -> `evaluation/results/nn_ens23.csv`, sha256
  `54711f5682b3f5e36394b770fe9c7ca64900a1af11a48106916d7c8cafd379d1`.
- Previous primary run2 alone: `nn_run2.csv`, sha256
  `c4f814f330c420684659c3d952cf201755a3da221d683068edb06ce15034dd29`.
- Weights sha256: run2 `best.pt` `e21cba8b90ed56710395ba14253ac33d2556f9530aba44899e3dc8c523f1c684`,
  run3 `best.pt` `169ce493c27aa2be45c38b6e96974bc4edb5be2eb437b115c945eb1d4eeb0ad3`.
- run3 alone (`--rand v2 --L 1500`) failed the rule (long-gap specialist, worse mean).
- run4 (`--rand mix --L 1500 --steps 300`) is prepared but on hold; it would be judged against
  the ensemble with the same rule.
- Dropped: GBDT path (collapses on ATD).

## Setting up on a new machine
1. Python 3.12.3, `python -m venv .venv`, `pip install -r requirements-mtah.txt`
   (torch CPU: `--index-url https://download.pytorch.org/whl/cpu`).
2. Competition data into `data/` (layout in `data/README.md`) from the organisers' download.
3. `cache/idsse_*.npz` from Drive `MyDrive/Metrica-Hackathon/mtah/cache/` (or re-download IDSSE
   per `external/README.md` and convert). `cache/BAR-ATH_tactical_atd.npz` rebuilds itself.
4. Model folders into `outputs/nn/run{1,2,3}/` (best.pt, decoder.json, val_probs.npz).
5. Check: `python -m pytest tests -q`, then
   `python -m mtah.make_submission --model outputs/nn/run2 outputs/nn/run3 --name nn_ens23 --refresh`
   and compare the sha256 above. Same hash = the pipeline reproduces on the new machine (this is
   also the reproducibility claim the repo must make). If it differs, compare the class shares and
   possession count first (H 37.4 / A 26.2 / D 36.4%, 306 possessions) before anything else.

## Next steps
1. Reproduce the ensemble hash on the new machine (above).
2. Optional run4 on Colab, judged against the ensemble by the same rule.
3. Final weights + decoder.json published with the repo (GitHub Release; `outputs/` is gitignored).
4. Technical doc (start ~14-15 Nov), submit via the Form, make the repo public.
5. Optional: Smart Tagging as extra evidence; own ball detection from the video (high effort).
