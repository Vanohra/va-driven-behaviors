"""
Emotion Analyzer Module

This module provides comprehensive emotion analysis functionality including:
- Calibration loading and management
- Trend analysis from time series data
- VA state label classification
- Robot reaction recommendations

All threshold calculations use calibration statistics when available.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import robust statistics utilities
from .robust_stats import (
    mad, iqr, trimmed_mean, winsorize, hampel_filter, downsample_series
)


# ============================================================================
# Calibration Management
# ============================================================================

def compute_calibration_stats(results: List[Dict]) -> Optional[Dict]:
    """
    Compute calibration statistics (min/max/mean/std/percentiles/MAD/IQR) from all video results.
    
    Args:
        results: List of result dictionaries with 'valence' and 'arousal' keys
    
    Returns:
        Dictionary with calibration stats for valence and arousal, including robust statistics
    """
    if not results:
        return None
    
    # Extract all valence and arousal values
    valence_values = [r['valence'] for r in results if 'valence' in r and 'error' not in r]
    arousal_values = [r['arousal'] for r in results if 'arousal' in r and 'error' not in r]
    
    if not valence_values or not arousal_values:
        return None
    
    valence_array = np.array(valence_values)
    arousal_array = np.array(arousal_values)
    
    # Compute percentiles
    percentiles = [10, 25, 30, 50, 70, 75, 90]
    
    def compute_stats(arr, name):
        stats_dict = {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'mad': float(mad(arr)),  # Median Absolute Deviation
            'iqr': float(iqr(arr))    # Interquartile Range
        }
        for p in percentiles:
            stats_dict[f'p{p}'] = float(np.percentile(arr, p))
        return stats_dict
    
    calibration = {
        'valence': compute_stats(valence_array, 'valence'),
        'arousal': compute_stats(arousal_array, 'arousal')
    }
    
    return calibration


def print_calibration_stats(calibration: Optional[Dict]):
    """Print calibration statistics in a readable format."""
    if not calibration:
        return
    
    print("=" * 60)
    print("VA CALIBRATION STATISTICS")
    print("=" * 60)
    print()
    
    for name in ['valence', 'arousal']:
        stats = calibration[name]
        print(f"{name.upper()}:")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Median: {stats['median']:.4f}")
        print(f"  Percentiles:")
        print(f"    P10: {stats['p10']:.4f}, P25: {stats['p25']:.4f}, P30: {stats['p30']:.4f}")
        print(f"    P50: {stats['p50']:.4f}, P70: {stats['p70']:.4f}, P75: {stats['p75']:.4f}, P90: {stats['p90']:.4f}")
        print()
    
    print("=" * 60)
    print()


def load_calibration(calibration_path: str) -> Optional[Dict]:
    """
    Load calibration statistics from a JSON file.
    
    Args:
        calibration_path: Path to calibration.json file
    
    Returns:
        Dictionary with calibration stats or None if file not found/invalid
    """
    calibration_path = Path(calibration_path)
    
    if not calibration_path.exists():
        print(f"Warning: Calibration file not found: {calibration_path}")
        print("Using fallback fixed thresholds.")
        return None
    
    try:
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)
        
        # Validate structure (be resilient to missing stats)
        required_keys = ['valence', 'arousal']
        
        # Absolute minimum needed for basic operation
        essential_stats = ['mean', 'median']
        
        for key in required_keys:
            if key not in calibration:
                print(f"Warning: Missing '{key}' in calibration. Using fallback thresholds.")
                return None
            
            # Ensure essential stats exist
            for stat in essential_stats:
                if stat not in calibration[key]:
                    print(f"Warning: Missing essential '{stat}' for '{key}'. Using fallbacks.")
                    return None
            
            # --- Smart Fill: Estimate missing stats from existing data ---
            
            # 1. Standard Deviation (std)
            if 'std' not in calibration[key]:
                # If we have percentiles, estimate std (Normal dist approx: p70-p30 is ~1.05 std)
                if 'p70' in calibration[key] and 'p30' in calibration[key]:
                    est_std = (calibration[key]['p70'] - calibration[key]['p30']) / 1.05
                    calibration[key]['std'] = max(0.01, float(est_std))
                else:
                    calibration[key]['std'] = 0.15 # Safe generic default
                print(f"  Note: Missing 'std' for '{key}' — estimated as {calibration[key]['std']:.4f}")

            # 2. Min/Max
            if 'min' not in calibration[key]: calibration[key]['min'] = -1.0
            if 'max' not in calibration[key]: calibration[key]['max'] = 1.0
            
            # 3. Percentiles (needed for the state labels)
            # Default spread if missing
            defaults = {
                'p10': -0.4, 'p25': -0.2, 'p30': -0.15, 'p50': 0.0, 
                'p70': 0.15, 'p75': 0.2, 'p90': 0.4
            }
            for p_key, p_val in defaults.items():
                if p_key not in calibration[key]:
                    # Shift default relative to the mean
                    calibration[key][p_key] = calibration[key]['mean'] + p_val

            # 4. MAD/IQR (Robust stats)
            if 'mad' not in calibration[key]:
                calibration[key]['mad'] = calibration[key]['std'] * 0.6745
            if 'iqr' not in calibration[key]:
                calibration[key]['iqr'] = calibration[key]['std'] * 1.349
        
        print(f"Calibration loaded successfully (with recovery) from: {calibration_path}")
        return calibration
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in calibration file: {e}")
        print("Using fallback fixed thresholds.")
        return None
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        print("Using fallback fixed thresholds.")
        return None


# ============================================================================
# Robust Preprocessing
# ============================================================================

def preprocess_series(x: np.ndarray, 
                     config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess a time series to handle outliers and noise.
    
    Steps:
    1. Convert to numpy float array, drop NaNs/inf
    2. Winsorize (cap extreme values)
    3. Apply Hampel filter (replace outliers with rolling median)
    4. Apply light smoothing (rolling median or EMA)
    
    Args:
        x: Input time series (1D numpy array)
        config: Optional configuration dict with:
            - winsor_limits: Tuple (lower, upper) quantiles (default: (0.01, 0.01))
            - hampel_window: Window size for Hampel filter (default: 7)
            - hampel_n_sigmas: Number of sigmas for outlier threshold (default: 3.0)
            - smooth_window: Window size for rolling median smoothing (default: 5)
            - smooth_method: 'rolling_median' or 'ema' (default: 'rolling_median')
            - ema_alpha: Alpha for EMA if method='ema' (default: 0.2)
    
    Returns:
        Tuple of (preprocessed_array, metadata_dict) where metadata contains:
            - outlier_rate: Fraction of points replaced by Hampel filter or clipped by winsorization
            - n_winsorized: Number of points clipped by winsorization
            - n_hampel_replaced: Number of points replaced by Hampel filter
    """
    if config is None:
        config = {}
    
    # Default configuration
    winsor_limits = config.get('winsor_limits', (0.01, 0.01))
    hampel_window = config.get('hampel_window', 7)
    hampel_n_sigmas = config.get('hampel_n_sigmas', 3.0)
    smooth_window = config.get('smooth_window', 5)
    smooth_method = config.get('smooth_method', 'rolling_median')
    ema_alpha = config.get('ema_alpha', 0.2)
    
    # Convert to numpy float array
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    # Drop NaNs/inf (keep valid indices for tracking)
    valid_mask = np.isfinite(x_flat)
    n_total = len(x_flat)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        # All invalid, return zeros
        return np.zeros_like(x), {'outlier_rate': 1.0, 'n_winsorized': 0, 'n_hampel_replaced': 0}
    
    # Work with valid data
    x_valid = x_flat[valid_mask].copy()
    
    # Track original for winsorization counting
    x_before_winsor = x_valid.copy()
    
    # Step 1: Winsorize
    x_winsorized = winsorize(x_valid, limits=winsor_limits)
    n_winsorized = np.sum(x_winsorized != x_before_winsor)
    
    # Step 2: Hampel filter (only if window size is reasonable)
    if len(x_winsorized) >= hampel_window:
        x_filtered, n_hampel_replaced = hampel_filter(
            x_winsorized, window=hampel_window, n_sigmas=hampel_n_sigmas
        )
    else:
        x_filtered = x_winsorized.copy()
        n_hampel_replaced = 0
    
    # Step 3: Light smoothing
    if len(x_filtered) >= smooth_window:
        if smooth_method == 'rolling_median':
            # Rolling median smoothing
            x_smoothed = np.zeros_like(x_filtered)
            half_window = smooth_window // 2
            for i in range(len(x_filtered)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(x_filtered), i + half_window + 1)
                x_smoothed[i] = np.median(x_filtered[start_idx:end_idx])
        elif smooth_method == 'ema':
            # Exponential moving average
            x_smoothed = np.zeros_like(x_filtered)
            x_smoothed[0] = x_filtered[0]
            for i in range(1, len(x_filtered)):
                x_smoothed[i] = ema_alpha * x_filtered[i] + (1 - ema_alpha) * x_smoothed[i-1]
        else:
            x_smoothed = x_filtered.copy()
    else:
        x_smoothed = x_filtered.copy()
    
    # Reconstruct full array (restore invalid positions as NaN)
    x_result = np.full(n_total, np.nan, dtype=np.float64)
    x_result[valid_mask] = x_smoothed
    
    # Restore original shape
    x_result = x_result.reshape(original_shape)
    
    # Compute outlier rate
    outlier_rate = (n_winsorized + n_hampel_replaced) / n_valid if n_valid > 0 else 0.0
    
    metadata = {
        'outlier_rate': float(outlier_rate),
        'n_winsorized': int(n_winsorized),
        'n_hampel_replaced': int(n_hampel_replaced)
    }
    
    return x_result, metadata


# ============================================================================
# Robust Baseline Computation
# ============================================================================

def compute_robust_baseline(x: np.ndarray,
                           preprocessed_x: Optional[np.ndarray] = None,
                           trim_ratio: float = 0.10,
                           outlier_rate: float = 0.0) -> Dict:
    """
    Compute robust session baseline from a time series.
    
    Uses trimmed mean as primary baseline estimator, with dynamic trim ratio
    based on outlier rate. Also computes median, MAD, and IQR for reliability metrics.
    
    Args:
        x: Original time series (1D numpy array)
        preprocessed_x: Preprocessed series (if None, uses x)
        trim_ratio: Base trim ratio (default: 0.10, i.e., 10% from each tail)
        outlier_rate: Outlier rate from preprocessing (used to adjust trim_ratio)
    
    Returns:
        Dictionary with baseline statistics:
        {
            'value': float,              # trimmed mean (primary baseline)
            'median': float,
            'mad': float,
            'iqr': float,
            'outlier_rate': float,
            'stability_score': float     # 0..1, higher is more stable
        }
    """
    if preprocessed_x is None:
        preprocessed_x = x
    
    # Convert to numpy and remove NaNs
    x_clean = np.asarray(preprocessed_x, dtype=np.float64)
    x_clean = x_clean[~np.isnan(x_clean)]
    
    if len(x_clean) == 0:
        return {
            'value': 0.0,
            'median': 0.0,
            'mad': 0.0,
            'iqr': 0.0,
            'outlier_rate': outlier_rate,
            'stability_score': 0.0
        }
    
    # Dynamic trim ratio: increase if outlier rate is high
    if outlier_rate > 0.15:
        dynamic_trim = min(0.20, trim_ratio + 0.05)
    elif outlier_rate < 0.05 and len(x_clean) < 100:
        # Short clips with low outlier rate: use less trimming
        dynamic_trim = max(0.05, trim_ratio - 0.05)
    else:
        dynamic_trim = trim_ratio
    
    # Compute robust statistics
    baseline_value = trimmed_mean(x_clean, trim_ratio=dynamic_trim)
    baseline_median = float(np.median(x_clean))
    baseline_mad_value = mad(x_clean)
    baseline_iqr_value = iqr(x_clean)
    
    # Compute stability score (0..1)
    # Lower volatility (MAD) relative to IQR suggests stability
    # Also penalize high outlier rate
    if baseline_iqr_value > 0:
        volatility_ratio = baseline_mad_value / (baseline_iqr_value / 1.349)  # Normalize IQR to std
        stability_score = max(0.0, min(1.0, 1.0 - min(volatility_ratio, 2.0) / 2.0))
    else:
        stability_score = 1.0 if baseline_mad_value < 1e-6 else 0.5
    
    # Penalize outlier rate
    stability_score = stability_score * (1.0 - min(outlier_rate, 0.5))
    
    return {
        'value': float(baseline_value),
        'median': float(baseline_median),
        'mad': float(baseline_mad_value),
        'iqr': float(baseline_iqr_value),
        'outlier_rate': float(outlier_rate),
        'stability_score': float(stability_score)
    }


# ============================================================================
# Trend Analysis Functions
# ============================================================================

def compute_robust_slope(x: np.ndarray, max_points: int = 500) -> Dict:
    """
    Compute robust slope using Theil-Sen estimator or fallback.
    
    Args:
        x: Time series (1D numpy array)
        max_points: Maximum points to use for slope computation (downsample if needed)
    
    Returns:
        Dictionary with:
        {
            'slope': float,
            'method': str,  # 'theil_sen' | 'ols_downsampled' | 'fallback'
            'n_used': int
        }
    """
    x_clean = np.asarray(x, dtype=np.float64)
    x_clean = x_clean[~np.isnan(x_clean)]
    
    if len(x_clean) < 2:
        return {'slope': 0.0, 'method': 'fallback', 'n_used': len(x_clean)}
    
    n = len(x_clean)
    indices = np.arange(n)
    
    # Downsample if needed
    if n > max_points:
        x_downsampled = downsample_series(x_clean, max_points=max_points)
        indices_downsampled = np.linspace(0, n - 1, len(x_downsampled), dtype=int)
        x_used = x_downsampled
        indices_used = indices_downsampled
        n_used = len(x_downsampled)
    else:
        x_used = x_clean
        indices_used = indices
        n_used = n
    
    # Try Theil-Sen estimator (robust to outliers)
    if HAS_SCIPY and n_used >= 2:
        try:
            # Theil-Sen estimator: median of all pairwise slopes
            result = stats.theilslopes(x_used, indices_used)
            slope = float(result.slope)
            method = 'theil_sen'
        except Exception:
            # Fallback to OLS on downsampled data
            if n_used >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(indices_used, x_used)
                slope = float(slope)
                method = 'ols_downsampled'
            else:
                slope = 0.0
                method = 'fallback'
    else:
        # Fallback: simple OLS or first-last approximation
        if n_used >= 2:
            slope = float((x_used[-1] - x_used[0]) / (indices_used[-1] - indices_used[0]))
            method = 'ols_downsampled'
        else:
            slope = 0.0
            method = 'fallback'
    
    return {
        'slope': slope,
        'method': method,
        'n_used': n_used
    }


def compute_robust_delta(x: np.ndarray,
                        window_proportion: float = 0.2,
                        min_window: int = 15,
                        max_window: int = 300) -> Dict:
    """
    Compute robust start/end delta using median windows with adaptive sizing.
    
    Args:
        x: Time series (1D numpy array)
        window_proportion: Proportion of series to use for start/end windows (default: 0.2)
        min_window: Minimum window size (default: 15)
        max_window: Maximum window size (default: 300)
    
    Returns:
        Dictionary with:
        {
            'delta': float,           # median(last_window) - median(first_window)
            'start_median': float,
            'end_median': float,
            'window_size': int
        }
    """
    x_clean = np.asarray(x, dtype=np.float64)
    x_clean = x_clean[~np.isnan(x_clean)]
    
    if len(x_clean) == 0:
        return {
            'delta': 0.0,
            'start_median': 0.0,
            'end_median': 0.0,
            'window_size': 0
        }
    
    n = len(x_clean)
    
    # Compute adaptive window size
    w = int(n * window_proportion)
    # Clamp: ensure window doesn't exceed half the series, and use at least 1 if series is very short
    w = max(1, min(w, max_window, n // 2, n))  # Clamp and ensure we don't exceed series length
    # But still respect min_window if series is long enough
    if n >= min_window * 2:
        w = max(min_window, w)
    
    if w == 0 or n < 2:
        return {
            'delta': 0.0,
            'start_median': float(np.median(x_clean)) if len(x_clean) > 0 else 0.0,
            'end_median': float(np.median(x_clean)) if len(x_clean) > 0 else 0.0,
            'window_size': w
        }
    
    # Compute medians of first and last windows
    start_window = x_clean[:w]
    end_window = x_clean[-w:]
    
    start_median = float(np.median(start_window))
    end_median = float(np.median(end_window))
    delta = end_median - start_median
    
    return {
        'delta': float(delta),
        'start_median': float(start_median),
        'end_median': float(end_median),
        'window_size': int(w)
    }


def compute_trends(time_series: np.ndarray, 
                   trend_threshold: Optional[float] = None,
                   calibration: Optional[Dict] = None,
                   preprocessed_series: Optional[np.ndarray] = None) -> Dict:
    """
    Compute robust trend statistics from a time series.
    
    Uses robust slope (Theil-Sen) and robust delta (median windows) with adaptive thresholds.
    
    Args:
        time_series: numpy array of values over time (original, for backward compatibility)
        trend_threshold: threshold for determining direction (if None, uses scale-aware default)
        calibration: Dictionary with calibration stats for scale-aware thresholds
        preprocessed_series: Optional preprocessed series (if None, uses time_series)
    
    Returns:
        Dictionary with trend statistics:
        - start_mean: median of first window (for backward compatibility)
        - end_mean: median of last window (for backward compatibility)
        - delta: robust delta (median-based)
        - slope: robust slope (Theil-Sen or fallback)
        - volatility: MAD of preprocessed series
        - direction: 'rising', 'falling', 'stable', 'mixed', or 'uncertain'
        - confidence: float (0..1)
        - window_size: int
        - slope_meta: dict with slope method info
    """
    if len(time_series) == 0:
        return {
            'start_mean': 0.0,
            'end_mean': 0.0,
            'delta': 0.0,
            'slope': 0.0,
            'volatility': 0.0,
            'direction': 'stable',
            'confidence': 0.0,
            'window_size': 0,
            'slope_meta': {}
        }
    
    # Use preprocessed series if available, otherwise use original
    series_for_trend = preprocessed_series if preprocessed_series is not None else time_series
    
    # Ensure 1D array
    if series_for_trend.ndim > 1:
        series_for_trend = series_for_trend.flatten()
    if time_series.ndim > 1:
        time_series = time_series.flatten()
    
    # Compute robust delta
    delta_info = compute_robust_delta(series_for_trend)
    delta = delta_info['delta']
    start_median = delta_info['start_median']
    end_median = delta_info['end_median']
    window_size = delta_info['window_size']
    
    # For backward compatibility, also compute means
    n = len(series_for_trend)
    start_window_mean = float(np.mean(series_for_trend[:min(window_size, n)])) if n > 0 else 0.0
    end_window_mean = float(np.mean(series_for_trend[-min(window_size, n):])) if n > 0 else 0.0
    
    # Compute robust slope
    slope_info = compute_robust_slope(series_for_trend, max_points=500)
    slope = slope_info['slope']
    
    # Compute volatility (MAD of preprocessed series)
    series_clean = series_for_trend[~np.isnan(series_for_trend)]
    if len(series_clean) > 0:
        volatility = mad(series_clean)
    else:
        volatility = 0.0
    
    # Compute scale-aware threshold using robust statistics
    if trend_threshold is None:
        if calibration:
            # Use robust scale (MAD) if available, else fall back to std
            v_scale = calibration['valence'].get('mad', calibration['valence'].get('std', 0.1))
            a_scale = calibration['arousal'].get('mad', calibration['arousal'].get('std', 0.1))
            avg_robust_scale = (v_scale + a_scale) / 2
            trend_threshold = max(0.001, 0.3 * avg_robust_scale)
        else:
            # Estimate from current clip's MAD
            if len(series_clean) > 0:
                clip_mad = mad(series_clean)
                trend_threshold = max(0.01, 0.3 * clip_mad)
            else:
                trend_threshold = 0.01
    
    # Determine direction with agreement check
    slope_sign = 1 if slope > 0 else (-1 if slope < 0 else 0)
    delta_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
    
    # Check magnitudes
    slope_significant = abs(slope) > trend_threshold
    delta_significant = abs(delta) > trend_threshold
    
    # High volatility suggests uncertainty
    volatility_threshold = trend_threshold * 2.0  # Rough heuristic
    high_volatility = volatility > volatility_threshold
    
    # Determine direction and confidence
    if slope_sign == delta_sign and slope_significant and delta_significant:
        # Agreement: clear trend
        if slope_sign > 0:
            direction = 'rising'
        elif slope_sign < 0:
            direction = 'falling'
        else:
            direction = 'stable'
        confidence = 0.8 + 0.2 * (1.0 - min(volatility / (volatility_threshold * 2), 1.0))
    elif slope_sign != delta_sign or (not slope_significant and not delta_significant):
        # Disagreement or both insignificant
        if high_volatility:
            direction = 'uncertain'
            confidence = 0.3
        else:
            direction = 'mixed'
            confidence = 0.5
    else:
        # One is significant, other is not
        if slope_significant:
            direction = 'rising' if slope_sign > 0 else 'falling'
        else:
            direction = 'rising' if delta_sign > 0 else 'falling'
        confidence = 0.6
    
    # Penalize confidence for high volatility and outlier rate
    if high_volatility:
        confidence *= 0.7
    
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        'start_mean': start_window_mean,  # For backward compatibility
        'end_mean': end_window_mean,      # For backward compatibility
        'delta': float(delta),
        'slope': float(slope),
        'volatility': float(volatility),
        'direction': direction,
        'confidence': float(confidence),
        'window_size': int(window_size),
        'slope_meta': slope_info
    }


# ============================================================================
# State Label Classification
# ============================================================================

def compute_state_label(valence_mean: float, 
                       arousal_mean: float,
                       calibration: Optional[Dict] = None,
                       debug: bool = False) -> str:
    """
    Compute a state label from baseline valence and arousal values using percentile-based thresholds.
    
    Args:
        valence_mean: Mean valence value (or average of start/end means)
        arousal_mean: Mean arousal value (or average of start/end means)
        calibration: Dictionary with calibration stats (percentiles) for valence and arousal.
                     If None, uses fixed thresholds (fallback).
        debug: If True, print debug information about thresholds used
    
    Returns:
        String state label: 'negative-high-arousal', 'negative-low-arousal', 
        'positive-high-arousal', 'positive-low-arousal', 'neutral', etc.
    """
    if calibration is None:
        # Fallback to fixed thresholds if no calibration available
        v_p30, v_p70 = -0.1, 0.1
        a_p30, a_p70 = -0.1, 0.1
        threshold_source = "fallback (fixed)"
    else:
        # Use percentile-based thresholds
        v_p30 = calibration['valence']['p30']
        v_p70 = calibration['valence']['p70']
        a_p30 = calibration['arousal']['p30']
        a_p70 = calibration['arousal']['p70']
        threshold_source = "calibration (percentiles)"
    
    if debug:
        print(f"    [DEBUG] State label computation:")
        print(f"      Thresholds ({threshold_source}): V_P30={v_p30:.4f}, V_P70={v_p70:.4f}, A_P30={a_p30:.4f}, A_P70={a_p70:.4f}")
        print(f"      Values: valence={valence_mean:.4f}, arousal={arousal_mean:.4f}")
    
    # Determine valence category using percentiles
    if valence_mean < v_p30:
        v_cat = 'negative'
    elif valence_mean > v_p70:
        v_cat = 'positive'
    else:
        v_cat = 'neutral-valence'
    
    # Determine arousal category using percentiles
    if arousal_mean > a_p70:
        a_cat = 'high-arousal'
    elif arousal_mean < a_p30:
        a_cat = 'low-arousal'
    else:
        a_cat = 'neutral-arousal'
    
    # Combine into state label
    if v_cat == 'neutral-valence' and a_cat == 'neutral-arousal':
        state_label = 'neutral'
    elif v_cat == 'neutral-valence':
        state_label = a_cat  # e.g., 'high-arousal' or 'low-arousal'
    elif a_cat == 'neutral-arousal':
        # To match the mapper's expected 8-state model, 
        # append 'low-arousal' to significant valence
        state_label = f"{v_cat}-low-arousal"
    else:
        state_label = f"{v_cat}-{a_cat}"  # e.g., 'negative-high-arousal'
    
    if debug:
        print(f"      State label: {state_label} (V={v_cat}, A={a_cat})")
    
    return state_label


# ============================================================================
# Reaction Recommendation
# ============================================================================

def recommend_reaction(valence_trend: Dict,
                      arousal_trend: Dict,
                      valence_mean: float,
                      arousal_mean: float,
                      volatility_threshold: Optional[float] = None,
                      calibration: Optional[Dict] = None,
                      debug: bool = False) -> Tuple[str, str]:
    """
    Recommend a robot reaction based on baseline VA state and trends.
    
    Primary reaction is determined by STATE (baseline VA quadrant/level),
    then modified by TREND (direction and volatility) for intensity.
    
    Args:
        valence_trend: Dictionary with valence trend stats (from compute_trends)
        arousal_trend: Dictionary with arousal trend stats (from compute_trends)
        valence_mean: Mean valence value (baseline state)
        arousal_mean: Mean arousal value (baseline state)
        volatility_threshold: threshold for high volatility (if None, uses scale-aware default)
        calibration: Dictionary with calibration stats (for state label computation)
        debug: If True, print debug information
    
    Returns:
        Tuple: (reaction_recommendation: str, notes: str)
    """
    v_dir = valence_trend['direction']
    a_dir = arousal_trend['direction']
    v_vol = valence_trend['volatility']
    a_vol = arousal_trend['volatility']
    v_delta = valence_trend['delta']
    a_delta = arousal_trend['delta']
    
    # Use scale-aware volatility threshold if not provided
    if volatility_threshold is None:
        if calibration:
            # Use 75th percentile of volatility as threshold
            # Estimate from std devs (volatility is std dev of time series)
            v_std = calibration['valence']['std']
            a_std = calibration['arousal']['std']
            volatility_threshold = max(v_std, a_std) * 1.5
        else:
            volatility_threshold = 0.25  # Fallback
    
    # Check for high volatility (uncertain state) - highest priority
    high_volatility = (v_vol > volatility_threshold) or (a_vol > volatility_threshold)
    
    if high_volatility:
        return (
            "Uncertain state: choose safest conservative behavior (slow/stop and increase personal space)",
            "High volatility detected"
        )
    
    # Compute state label from baseline VA (uses calibration percentiles)
    state_label = compute_state_label(valence_mean, arousal_mean, calibration=calibration, debug=debug)
    
    # Determine base reaction from STATE
    # Then modify intensity based on TREND
    
    # NEGATIVE-HIGH-AROUSAL (Angry, Fearful, etc.) - Safety priority
    if state_label == 'negative-high-arousal':
        if v_dir == 'falling' or a_dir == 'rising':
            # Escalating: stronger de-escalation
            return (
                "De-escalate strongly: stop immediately, increase distance significantly, reduce all stimuli, use calm voice/LED",
                f"Negative-high-arousal state with escalating trends (V={v_dir}, A={a_dir})"
            )
        elif v_dir == 'rising' or a_dir == 'falling':
            # Improving: moderate de-escalation
            return (
                "De-escalate: slow movement, increase distance, reduce stimuli, use calm voice/LED",
                f"Negative-high-arousal state with improving trends (V={v_dir}, A={a_dir})"
            )
        else:
            # Stable negative-high: still need de-escalation
            return (
                "De-escalate: maintain distance, minimize movement, use calm voice/LED",
                "Stable negative-high-arousal state (consistently negative/high)"
            )
    
    # NEGATIVE-LOW-AROUSAL (Sad, Depressed) - Check-in priority
    elif state_label == 'negative-low-arousal':
        if v_dir == 'falling':
            # Worsening: stronger check-in
            return (
                "Check-in actively: slow down, keep distance, offer help, minimize movement, gentle tone",
                f"Negative-low-arousal state with worsening valence (V={v_dir})"
            )
        elif v_dir == 'rising':
            # Improving: supportive presence
            return (
                "Supportive presence: maintain calm distance, gentle interaction, offer help if appropriate",
                f"Negative-low-arousal state with improving valence (V={v_dir})"
            )
        else:
            # Stable negative-low: check-in
            return (
                "Check-in: slow down, keep distance, offer help, minimize movement",
                "Stable negative-low-arousal state (consistently negative/low)"
            )
    
    # POSITIVE-HIGH-AROUSAL (Happy, Excited) - Engage
    elif state_label == 'positive-high-arousal':
        if v_dir == 'rising' and a_dir == 'rising':
            # Escalating positive: full engagement
            return (
                "Engage fully: maintain interaction, normal speed, playful/encouraging tone",
                f"Positive-high-arousal state with escalating trends (V={v_dir}, A={a_dir})"
            )
        elif v_dir == 'falling' or a_dir == 'falling':
            # Decreasing: monitor
            return (
                "Engage with monitoring: maintain interaction, watch for changes",
                f"Positive-high-arousal state with decreasing trends (V={v_dir}, A={a_dir})"
            )
        else:
            # Stable positive-high: engage
            return (
                "Engage: maintain interaction, normal speed, playful/encouraging tone",
                "Stable positive-high-arousal state (consistently positive/high)"
            )
    
    # POSITIVE-LOW-AROUSAL (Calm, Content) - Maintain
    elif state_label == 'positive-low-arousal':
        if v_dir == 'rising':
            # Improving: maintain with encouragement
            return (
                "Maintain with encouragement: continue task, calm presence, positive tone",
                f"Positive-low-arousal state with improving valence (V={v_dir})"
            )
        elif v_dir == 'falling':
            # Worsening: monitor
            return (
                "Maintain and monitor: continue task, watch for changes",
                f"Positive-low-arousal state with worsening valence (V={v_dir})"
            )
        else:
            # Stable positive-low: maintain
            return (
                "Maintain: continue task, calm presence",
                "Stable positive-low-arousal state (consistently positive/low)"
            )
    
    # NEUTRAL - Assess based on trends
    elif state_label == 'neutral':
        if a_dir == 'rising':
            # Rising arousal from neutral: caution
            return (
                "Caution: pause and assess, avoid sudden moves, monitor for escalation",
                f"Neutral state with rising arousal (A={a_dir})"
            )
        elif v_dir == 'falling' or a_dir == 'falling':
            # Negative trends: check-in
            return (
                "Check-in: monitor situation, maintain distance, offer help if needed",
                f"Neutral state with negative trends (V={v_dir}, A={a_dir})"
            )
        elif v_dir == 'rising':
            # Positive trends: engage
            return (
                "Engage: maintain interaction, normal behavior",
                f"Neutral state with positive trends (V={v_dir})"
            )
        else:
            # Stable neutral: continue
            return (
                "Continue: maintain current behavior and interaction level",
                "Stable neutral state"
            )
    
    # HIGH-AROUSAL (neutral valence, high arousal) - Caution
    elif state_label == 'high-arousal':
        if v_dir == 'falling':
            # Negative trend: de-escalate
            return (
                "De-escalate: increase distance, reduce stimuli, use calm voice/LED",
                f"High-arousal state with negative valence trend (V={v_dir})"
            )
        elif a_dir == 'rising':
            # Escalating arousal: strong caution
            return (
                "Caution strongly: pause and assess, avoid sudden moves, increase distance",
                f"High-arousal state with rising arousal (A={a_dir})"
            )
        else:
            # Stable high-arousal: caution
            return (
                "Caution: pause and assess, avoid sudden moves",
                "Stable high-arousal state (consistently high arousal)"
            )
    
    # LOW-AROUSAL (neutral valence, low arousal) - Calm presence
    elif state_label == 'low-arousal':
        if v_dir == 'falling':
            # Negative trend: check-in
            return (
                "Check-in: monitor situation, offer help if needed",
                f"Low-arousal state with negative valence trend (V={v_dir})"
            )
        else:
            # Stable low-arousal: calm presence
            return (
                "Calm presence: maintain current behavior, low activity",
                "Stable low-arousal state (consistently low arousal)"
            )
    
    # FALLBACK for any unhandled state
    else:
        return (
            "Assess: monitor situation, use conservative behavior",
            f"Unusual state: {state_label} with trends (V={v_dir}, A={a_dir})"
        )


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_emotion_stream(valence_series: np.ndarray,
                           arousal_series: np.ndarray,
                           calibration: Optional[Dict] = None,
                           trend_threshold: Optional[float] = None,
                           volatility_threshold: Optional[float] = None,
                           export_timeseries: bool = False,
                           debug: bool = False) -> Dict:
    """
    Analyze emotion from valence and arousal time series.
    
    This is the main function that processes emotion data and returns comprehensive analysis.
    
    Args:
        valence_series: numpy array of per-frame valence predictions
        arousal_series: numpy array of per-frame arousal predictions
        calibration: Dictionary with calibration stats (from load_calibration)
        trend_threshold: Optional threshold for trend direction (uses scale-aware default if None)
        volatility_threshold: Optional threshold for high volatility (uses scale-aware default if None)
        export_timeseries: If True, include full time series in output
        debug: If True, print debug information
    
    Returns:
        Dictionary with comprehensive emotion analysis:
        {
            'valence': float,
            'arousal': float,
            'valence_std': float,
            'arousal_std': float,
            'valence_direction': str,
            'arousal_direction': str,
            'valence_delta': float,
            'arousal_delta': float,
            'valence_slope': float,
            'arousal_slope': float,
            'valence_volatility': float,
            'arousal_volatility': float,
            'va_state_label': str,
            'reaction_recommendation': str,
            'notes': str,
            'valence_timeseries': list (optional),
            'arousal_timeseries': list (optional)
        }
    """
    # Ensure 1D arrays
    if valence_series.ndim > 1:
        valence_series = valence_series.flatten()
    if arousal_series.ndim > 1:
        arousal_series = arousal_series.flatten()
    
    # Validate lengths match
    if len(valence_series) != len(arousal_series):
        raise ValueError(f"Valence and arousal series must have same length. Got {len(valence_series)} and {len(arousal_series)}")
    
    if len(valence_series) == 0:
        raise ValueError("Empty time series provided")
    
    # Robust preprocessing configuration
    preprocess_config = {
        'winsor_limits': (0.01, 0.01),
        'hampel_window': 7,
        'hampel_n_sigmas': 3.0,
        'smooth_window': 5,
        'smooth_method': 'rolling_median'
    }
    
    # Preprocess both series
    valence_preprocessed, valence_preprocess_meta = preprocess_series(valence_series, preprocess_config)
    arousal_preprocessed, arousal_preprocess_meta = preprocess_series(arousal_series, preprocess_config)
    
    # Compute robust baselines
    valence_baseline = compute_robust_baseline(
        valence_series, 
        preprocessed_x=valence_preprocessed,
        trim_ratio=0.10,
        outlier_rate=valence_preprocess_meta['outlier_rate']
    )
    arousal_baseline = compute_robust_baseline(
        arousal_series,
        preprocessed_x=arousal_preprocessed,
        trim_ratio=0.10,
        outlier_rate=arousal_preprocess_meta['outlier_rate']
    )
    
    # Extract baseline values (for backward compatibility and state labeling)
    mean_valence = valence_baseline['value']  # trimmed mean
    mean_arousal = arousal_baseline['value']  # trimmed mean
    
    # Also compute std for backward compatibility
    std_valence = float(np.std(valence_preprocessed[~np.isnan(valence_preprocessed)]))
    std_arousal = float(np.std(arousal_preprocessed[~np.isnan(arousal_preprocessed)]))
    
    # Compute robust trends (using preprocessed series)
    valence_trend = compute_trends(
        valence_series, 
        trend_threshold, 
        calibration,
        preprocessed_series=valence_preprocessed
    )
    arousal_trend = compute_trends(
        arousal_series, 
        trend_threshold, 
        calibration,
        preprocessed_series=arousal_preprocessed
    )
    
    # Compute state label from robust baseline VA (uses percentile-based thresholds from calibration)
    state_label = compute_state_label(mean_valence, mean_arousal, calibration, debug=debug)
    
    # Get reaction recommendation (uses both state and trends, with scale-aware thresholds)
    reaction, notes = recommend_reaction(
        valence_trend, arousal_trend, mean_valence, mean_arousal,
        volatility_threshold, calibration, debug=debug
    )
    
    # Build result dictionary (maintain backward compatibility)
    result = {
        'valence': mean_valence,
        'arousal': mean_arousal,
        'valence_std': std_valence,
        'arousal_std': std_arousal,
        'valence_direction': valence_trend['direction'],
        'arousal_direction': arousal_trend['direction'],
        'valence_delta': valence_trend['delta'],
        'arousal_delta': arousal_trend['delta'],
        'valence_slope': valence_trend['slope'],
        'arousal_slope': arousal_trend['slope'],
        'valence_volatility': valence_trend['volatility'],
        'arousal_volatility': arousal_trend['volatility'],
        'va_state_label': state_label,
        'reaction_recommendation': reaction,
        'notes': notes
    }
    
    # Add new reliability metadata (optional, additive)
    result['va_baseline'] = {
        'valence': valence_baseline,
        'arousal': arousal_baseline
    }
    result['trends'] = {
        'valence': {
            'confidence': valence_trend.get('confidence', 0.5),
            'window_size': valence_trend.get('window_size', 0),
            'slope_meta': valence_trend.get('slope_meta', {})
        },
        'arousal': {
            'confidence': arousal_trend.get('confidence', 0.5),
            'window_size': arousal_trend.get('window_size', 0),
            'slope_meta': arousal_trend.get('slope_meta', {})
        }
    }
    
    # Compute overall state confidence
    state_confidence = min(
        valence_baseline['stability_score'],
        arousal_baseline['stability_score'],
        (valence_trend.get('confidence', 0.5) + arousal_trend.get('confidence', 0.5)) / 2
    )
    result['state_confidence'] = float(state_confidence)
    
    # Optionally include full time series
    if export_timeseries:
        result['valence_timeseries'] = valence_series.tolist()
        result['arousal_timeseries'] = arousal_series.tolist()
    
    return result
