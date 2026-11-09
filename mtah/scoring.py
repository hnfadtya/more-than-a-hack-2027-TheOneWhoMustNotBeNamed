"""Score any pair of label arrays with the *official* metric code, unchanged.

The competition scorer only reads CSVs against the public window. For model selection we need the
same numbers on external matches (whose labels we derive ourselves), so this imports
evaluation/possession_metrics.py directly rather than re-implementing anything.
"""

import os
import sys

import numpy as np

from .common import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def score(gt_codes, pred_codes):
    """Official metric dict (spell_f1_mean, macro_f1, ...) for two int code arrays."""
    return pm.evaluate(np.asarray(gt_codes, dtype=np.int8), np.asarray(pred_codes, dtype=np.int8))


def brief(result):
    """One line, the numbers we actually look at."""
    return (f"spellF1 {result['spell_f1_mean']:.4f}  macroF1 {result['macro_f1']:.4f}  "
            f"poss {result['n_pred_spells']}/{result['n_gt_spells']}  "
            f"dur {result['mean_spell_duration_s']:.1f}s/{result['gt_mean_spell_duration_s']:.1f}s  "
            f"aliveF1 {result['alive_dead_f1']:.3f}")


def public_gt():
    """Codes of the public 5-minute ground truth. For pipeline checks ONLY -- never for tuning."""
    path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    return pm.label_codes(pm.read_label_csv(path))
