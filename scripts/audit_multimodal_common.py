"""Read-only audit metrics shared by the 2026-11-09 multimodal pairing audit.

Times are integer Unix seconds after each source reader applies its documented
clock convention. Naive clocks are NOT silently assigned the Windows timezone.
Five-minute occupancy is a screening estimate, not continuous valid wear time.
Expanding a CGM observation over its native cadence creates a coverage mask only;
it never creates/interpolates glucose targets. HR bins require >=1 observation.
"""
from __future__ import annotations

import math
import numpy as np

BIN_SECONDS = 300


def unique_seconds(values):
    a = np.asarray(values, dtype=np.float64)
    return np.unique(a[np.isfinite(a)].astype(np.int64))


def coverage_bins(seconds, cadence_minutes=5.0):
    """Forward native-cadence occupancy, rounded to whole five-minute bins."""
    t = unique_seconds(seconds)
    base = t // BIN_SECONDS
    n = max(1, int(math.ceil(float(cadence_minutes) / 5.0)))
    if n == 1:
        return np.unique(base)
    return np.unique(np.concatenate([base + k for k in range(n)]))


def pair_summary(cgm_bins, wearable_bins):
    cg = np.unique(np.asarray(cgm_bins, dtype=np.int64))
    hr = np.unique(np.asarray(wearable_bins, dtype=np.int64))
    common = np.intersect1d(cg, hr, assume_unique=True)
    gaps = np.diff(common)
    runs = np.split(common, np.where(gaps > 1)[0] + 1) if len(common) else []
    return {
        'cgm_occupied_5min_bins': int(len(cg)),
        'wearable_occupied_5min_bins': int(len(hr)),
        'paired_5min_bins': int(len(common)),
        'paired_person_hours': float(len(common) / 12),
        'fraction_cgm_coverage_with_wearable': float(len(common) / len(cg)) if len(cg) else None,
        'longest_paired_run_hours': float(max((len(x) for x in runs), default=0) / 12),
        'paired_run_count': int(len(runs)),
    }


def meal_window_audit(meal_seconds, cgm_seconds, hr_bins, cadence_minutes=5.0):
    """One row per exact-time meal event; do not use food-item rows as meals.

    Require >=80% expected *observed* CGM before/after meal separately, <=20min
    maximum gap including window edges; and >=5/6 complete premeal HR bins.
    Meal t need not lie on the grid: HR bins stop at floor(t/300)*300, ensuring
    no postmeal observations enter the premeal coverage decision. A future meal
    flag is label eligibility only and never a prospective predictor.
    """
    meals = unique_seconds(meal_seconds)
    cg = unique_seconds(cgm_seconds)
    hr = set(int(x) for x in hr_bins)
    cadence = float(cadence_minutes)
    pre_expected = int(math.ceil(30 / cadence))
    post_expected = int(math.ceil(120 / cadence))
    result = []
    for t in meals:
        lo, mid, hi = np.searchsorted(cg, [t - 1800, t, t + 7200])
        pre = cg[lo:mid]
        post = cg[mid:hi]
        observed = cg[lo:hi]
        edges = np.r_[t - 1800, observed, t + 7200]
        max_gap = int(np.diff(edges).max()) if len(edges) > 1 else 9000
        pre_hr = sum(int(t // 300) - k in hr for k in range(1, 7))
        cgm_ok = len(pre) >= math.ceil(.8 * pre_expected) and len(post) >= math.ceil(.8 * post_expected) and max_gap <= 1200
        later = meals[np.searchsorted(meals, t, side='right'):np.searchsorted(meals, t + 7200)]
        result.append({
            'meal_epoch_seconds': int(t),
            'cgm_pre_observed': int(len(pre)),
            'cgm_post_observed': int(len(post)),
            'cgm_pre_expected': pre_expected,
            'cgm_post_expected': post_expected,
            'cgm_max_gap_minutes': max_gap / 60,
            'premeal_hr_occupied_bins': pre_hr,
            'cgm_window_eligible': bool(cgm_ok),
            'cgm_and_premeal_hr_eligible': bool(cgm_ok and pre_hr >= 5),
            'another_logged_event_within_120min': bool(len(later)),
            'isolated_cgm_and_premeal_hr_eligible': bool(cgm_ok and pre_hr >= 5 and not len(later)),
        })
    return result


def merge_events_within_minutes(seconds, minutes=30):
    """Sensitivity only: start a new cluster when consecutive events >30min apart."""
    t = unique_seconds(seconds)
    if not len(t):
        return t
    return t[np.r_[True, np.diff(t) > minutes * 60]]


METRIC_NOTES = {
    'grid': '5min occupancy: HR >=1 valid observation; CGM each original observation covers its native cadence. Masks only, no glucose interpolation.',
    'paired_hours': 'Number of common occupied bins /12 summed within source subjects; not continuous duration or clinical completeness.',
    'meal_event': 'One distinct participant/start timestamp after exact record duplicate handling. Food lines sharing a timestamp are one event; 30min clustering is sensitivity, not proven biological meals.',
    'meal_window': '[-30,+120)min; observed CGM pre and post each >=80% native expected, max gap including boundaries <=20min; >=5/6 complete premeal HR bins.',
    'hr_range': '30–220 bpm is an audit plausibility screen, not a clinically validated cleaning rule; compare with all positive HR where possible.',
    'clock': 'Matching wall clocks is provisional if timezone/clock calibration is undocumented; no offset chosen by maximizing overlap.',
    'isolated_field_caveat': 'Legacy isolated_* names mean no later logged intake in the next120min only; they do not exclude prior meals, missing logs, or prove biological isolation.',
}
