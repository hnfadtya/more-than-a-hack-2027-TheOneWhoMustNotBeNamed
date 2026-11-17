# The One Who Must Not Be Named — More than a Hack 2027, Phase 1 submission

This repository regenerates our possession file for BAR-ATH, `The One Who Must Not Be Named.csv`.
How it works, what we tried and dropped, the training data and the limitations are in the technical
document, [docs/technical_report.pdf](docs/technical_report.pdf). The day-by-day log with every
validation number and the decision rules written before each result is [WORKLOG.md](WORKLOG.md).

In short: `PossessionNet` (set transformer + temporal convolutions + BiGRU, `mtah/model.py`) is
trained only on the open IDSSE matches (Bassek et al., CC BY 4.0), corrupted to look like the
competition's automatic tracking. Four trained networks are averaged and decoded with a Viterbi
decoder with minimum durations (`mtah/decode.py`). Ball gaps in the tracking are first filled by a
YOLO11s ball detector run on the tactical video (`mtah/vball.py`). No human label on BAR-ATH is
used anywhere.

The submitted file has this SHA-256:

```
dc7264ec037c74e49cf2a3d65cae5ed5131912dd15a33be268b6206de8f06054
```

## Reproduce the submission

1. **Environment.** Python 3.12.3.

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements-mtah.txt
   # CPU-only torch, if wanted: pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Match data.** Unpack the organisers' bundle into `data/` exactly as described in
   [data/README.md](data/README.md). The tactical video `data/tactical/BAR-ATH_tactical.mp4` is
   needed. The data is not in this repository and must not be.

3. **Weights.** `outputs/` is not committed. The four networks, the decoder settings and the ball
   detector are assets of the [v1.0 release](https://github.com/hnfadtya/more-than-a-hack-2027-TheOneWhoMustNotBeNamed/releases/tag/v1.0).
   This downloads them to the paths the pipeline reads and checks each SHA-256:

   ```bash
   python -m mtah.get_weights
   ```

   | release asset | goes to |
   |---|---|
   | `run2_best.pt`, `run3_best.pt`, `run5_best.pt`, `run6_best.pt` | `outputs/nn/run{2,3,5,6}/best.pt` |
   | `ens235_decoder.json` | `outputs/nn/ens235/decoder.json` |
   | `vball_ft1_best.pt` | `outputs/vball/ft1/best.pt` |

4. **Video ball.** Detect the ball on every graded frame where the tracking has none. This writes
   `cache/BAR-ATH_vball_ft1.npz`.

   ```bash
   python -m mtah.vball fill --tag ft1 --device mps   # Apple silicon, about 2.7 h
   python -m mtah.vball fill --tag ft1 --device cpu   # any machine, about 7.5 h
   ```

   CPU and Apple MPS give identical detections (batch 8, the default). **Do not run this step on a
   CUDA GPU**: the detections differ slightly and so does the final file.

5. **Possession file.** This runs on CPU and is deterministic.

   ```bash
   python -m mtah.make_submission \
       --model outputs/nn/run2 outputs/nn/run3 outputs/nn/run5 outputs/nn/run6 \
       --decoder outputs/nn/ens235/decoder.json \
       --vball ft1 --name nn_ens2356d_vb1
   ```

   It writes `evaluation/results/nn_ens2356d_vb1.csv` and prints the class shares
   (H 37.0% / A 24.9% / D 38.1% of graded frames, 331 possessions). The score it prints on the
   public five minutes is only a pipeline check.

6. **Name and check.**

   ```bash
   cp evaluation/results/nn_ens2356d_vb1.csv "The One Who Must Not Be Named.csv"
   python evaluation/validate_submission.py "The One Who Must Not Be Named.csv"
   shasum -a 256 "The One Who Must Not Be Named.csv"     # Windows: certutil -hashfile "The One Who Must Not Be Named.csv" SHA256
   ```

   The hash must be the one above.

Unit tests: `python -m pytest tests -q`.

**Training (not needed to reproduce).** IDSSE download and conversion are described in
[external/README.md](external/README.md). The networks were trained on a Colab A100 with
[notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) and the ball detector with
[notebooks/colab_vball.ipynb](notebooks/colab_vball.ipynb). The exact command of every run is in
WORKLOG.md.

---

*The organisers' original README follows.*

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
