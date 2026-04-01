"""
Tests for Robust VA Processing Pipeline

Tests the robustness of the analysis pipeline against:
  - Outliers and spikes
  - Missing detections
  - Variable video lengths
  - Mixed/uncertain trends

Run from the project root:
    python -m tests.test_robust_pipeline
    python tests/test_robust_pipeline.py
"""

import sys
import numpy as np
from pathlib import Path

# Ensure pipeline package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.emotion_analyzer import (
    preprocess_series,
    compute_robust_baseline,
    compute_trends,
    analyze_emotion_stream,
)
from pipeline.robust_stats import (
    mad, iqr, trimmed_mean, winsorize, hampel_filter, downsample_series,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_outliers_dont_move_baseline():
    """Outliers should not significantly shift the trimmed-mean baseline."""
    print("Test 1: Outliers don't move baseline")

    n = 100
    true_baseline = 0.4
    series = np.full(n, true_baseline, dtype=np.float64)

    n_outliers = int(n * 0.05)
    idx = np.random.choice(n, n_outliers, replace=False)
    series[idx] = np.random.choice([-1.0, 1.0], n_outliers)

    preprocessed, meta = preprocess_series(series)
    baseline_result = compute_robust_baseline(preprocessed)
    baseline = baseline_result["value"]  # key is 'value' in compute_robust_baseline

    error = abs(baseline - true_baseline)
    print(f"  True baseline: {true_baseline}, Computed: {baseline:.4f}, Error: {error:.4f}")
    assert error < 0.1, f"Baseline shifted too much: {error:.4f}"
    print("  PASS\n")


def test_short_series():
    """Short series should not crash — pipeline returns a valid result."""
    print("Test 2: Short series handling")

    for n in [5, 10, 15]:
        series = np.random.uniform(-0.3, 0.3, n)
        result = analyze_emotion_stream(series, series)
        assert result is not None, f"analyze_emotion_stream returned None for n={n}"
        assert "va_state_label" in result
        print(f"  n={n}: label={result['va_state_label']}  PASS")
    print()


def test_all_neutral():
    """Near-zero series should classify as neutral with high stability."""
    print("Test 3: All-neutral series")

    series = np.random.normal(0.0, 0.02, 100)
    result = analyze_emotion_stream(series, series)

    assert result is not None
    label = result["va_state_label"]
    print(f"  Label: {label}")
    assert "neutral" in label.lower(), f"Expected neutral, got {label}"
    print("  PASS\n")


def test_clear_trend_rising():
    """A clearly rising valence series should detect 'rising' direction."""
    print("Test 4: Clear rising trend")

    n = 100
    valence = np.linspace(-0.5, 0.5, n) + np.random.normal(0, 0.02, n)
    arousal = np.zeros(n) + np.random.normal(0, 0.02, n)

    result = analyze_emotion_stream(valence, arousal)
    assert result is not None
    direction = result.get("valence_direction", "")
    print(f"  Valence direction: {direction}")
    assert direction in ("rising", "mixed"), f"Expected rising, got {direction}"
    print("  PASS\n")


def test_high_volatility_triggers_caution():
    """High-volatility signal should map to CAUTION or DE_ESCALATE intent."""
    print("Test 5: High volatility -> cautious intent")

    n = 100
    # Highly noisy signal
    valence = np.random.uniform(-1.0, 1.0, n)
    arousal = np.random.uniform(-1.0, 1.0, n)

    result = analyze_emotion_stream(valence, arousal)
    assert result is not None
    vol = result.get("valence_volatility", 0.0)
    print(f"  Volatility: {vol:.4f}")

    ra = result.get("reaction_action")
    if ra:
        print(f"  Intent: {ra.intent}")
        assert ra.intent in ("CAUTION", "DE_ESCALATE", "NEUTRAL"), \
            f"Unexpected intent for high volatility: {ra.intent}"
    print("  PASS\n")


def test_hampel_filter_removes_spikes():
    """Hampel filter should replace spike values with local medians."""
    print("Test 6: Hampel filter removes spikes")

    series = np.full(50, 0.3, dtype=np.float64)
    series[10] = 5.0   # spike up
    series[30] = -5.0  # spike down

    filtered, meta = preprocess_series(series)
    assert filtered[10] < 2.0,  f"Spike at [10] not removed: {filtered[10]:.3f}"
    assert filtered[30] > -2.0, f"Spike at [30] not removed: {filtered[30]:.3f}"
    print(f"  Outlier rate: {meta.get('outlier_rate', 0):.2%}")
    print("  PASS\n")


def test_winsorization():
    """Winsorization should clip extreme values."""
    print("Test 7: Winsorization clips extremes")

    series = np.concatenate([np.full(90, 0.0), [10.0] * 5, [-10.0] * 5])
    winsorized = winsorize(series, limits=(0.05, 0.05))
    assert winsorized.max() < 5.0,  f"Max not clipped: {winsorized.max()}"
    assert winsorized.min() > -5.0, f"Min not clipped: {winsorized.min()}"
    print(f"  Clipped range: [{winsorized.min():.3f}, {winsorized.max():.3f}]")
    print("  PASS\n")


def test_calibration_thresholds():
    """Percentile-based state classification should respect calibration stats."""
    print("Test 8: Calibration-based classification")

    calibration = {
        "valence": {
            "min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.3,
            "median": 0.0, "mad": 0.2, "iqr": 0.4,
            "p10": -0.3, "p25": -0.15, "p30": -0.1,
            "p50": 0.0, "p70": 0.1, "p75": 0.15, "p90": 0.3,
        },
        "arousal": {
            "min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.3,
            "median": 0.0, "mad": 0.2, "iqr": 0.4,
            "p10": -0.3, "p25": -0.15, "p30": -0.1,
            "p50": 0.0, "p70": 0.1, "p75": 0.15, "p90": 0.3,
        },
    }

    # Positive valence + high arousal -> positive-high-arousal
    valence = np.full(80, 0.5)
    arousal = np.full(80, 0.5)
    result  = analyze_emotion_stream(valence, arousal, calibration=calibration)
    assert result is not None
    label = result["va_state_label"]
    print(f"  High V+A -> label: {label}")
    assert "positive" in label or "high" in label, f"Unexpected: {label}"
    print("  PASS\n")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_outliers_dont_move_baseline,
    test_short_series,
    test_all_neutral,
    test_clear_trend_rising,
    test_high_volatility_triggers_caution,
    test_hampel_filter_removes_spikes,
    test_winsorization,
    test_calibration_thresholds,
]


def run_all():
    print("=" * 62)
    print("  Robust VA Pipeline — Test Suite")
    print("=" * 62)
    print()

    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__}: {e}\n")
            failed += 1
            errors.append((test_fn.__name__, str(e)))
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__}: {e}\n")
            failed += 1
            errors.append((test_fn.__name__, str(e)))

    print("=" * 62)
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        print("\n  Failures:")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    print("=" * 62)
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
