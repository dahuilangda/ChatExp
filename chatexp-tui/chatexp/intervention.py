"""Observed-data admission rules; fixed utility defines campaign health."""
from __future__ import annotations
import numpy as np


def intervention_admission(history, completed, initial_count, batch_size, probes_spent, cooldown):
    """Intervene only after two BO batches, with a stalled low incumbent.

    Thresholds are frozen engineering rules, not learned probabilities.
    A nonimproving probe-containing batch pauses interventions for one round; at most two real probes.
    """
    if completed < initial_count + 2 * batch_size:
        return False, 'insufficient_sequential_evidence'
    if not history or not np.isfinite(history[-1]):
        return False, 'invalid_fixed_utility'
    if history[-1] >= .35:
        return False, 'healthy_fixed_incumbent'
    if history[-1] - history[-batch_size-1] > 1e-9:
        return False, 'engine_improving'
    if cooldown:
        return False, 'failed_probe_cooldown'
    if probes_spent >= 2:
        return False, 'probe_budget_exhausted'
    return True, 'stalled_low_fixed_incumbent'
