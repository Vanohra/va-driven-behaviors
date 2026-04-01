"""
Robust Statistics Utilities

This module provides robust statistical functions for handling outliers and noisy data
in time series analysis. These functions are designed to be resilient to outliers,
missing detections, and highly variable series lengths.

All functions use numpy as the base, with optional scipy enhancements where available.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def mad(x: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute Median Absolute Deviation (MAD).
    
    MAD is a robust measure of scale, defined as:
        MAD = median(|x_i - median(x)|)
    
    It is less sensitive to outliers than standard deviation.
    
    Args:
        x: Input array (1D)
        epsilon: Small value to guard against division by zero
    
    Returns:
        MAD value (float)
    """
    if len(x) == 0:
        return epsilon
    
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return epsilon
    
    median = np.median(x)
    deviations = np.abs(x - median)
    mad_value = np.median(deviations)
    
    return float(max(mad_value, epsilon))


def iqr(x: np.ndarray) -> float:
    """
    Compute Interquartile Range (IQR).
    
    IQR = Q3 - Q1, where Q3 is the 75th percentile and Q1 is the 25th percentile.
    This is a robust measure of spread that is insensitive to outliers.
    
    Args:
        x: Input array (1D)
    
    Returns:
        IQR value (float)
    """
    if len(x) == 0:
        return 0.0
    
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return 0.0
    
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    
    return float(q3 - q1)


def trimmed_mean(x: np.ndarray, trim_ratio: float = 0.10) -> float:
    """
    Compute trimmed mean (mean after removing extreme values from both tails).
    
    The trimmed mean removes trim_ratio proportion of values from both the lower
    and upper tails, then computes the mean of the remaining values. This makes
    it robust to outliers.
    
    Formula:
        Sort x: x_sorted = sort(x)
        n_trim = floor(len(x) * trim_ratio)
        trimmed_x = x_sorted[n_trim : len(x) - n_trim]
        trimmed_mean = mean(trimmed_x)
    
    Args:
        x: Input array (1D)
        trim_ratio: Proportion to trim from each tail (default: 0.10, i.e., 10% from each tail)
    
    Returns:
        Trimmed mean (float)
    """
    if len(x) == 0:
        return 0.0
    
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return 0.0
    
    # Clamp trim_ratio to valid range
    trim_ratio = max(0.0, min(0.5, trim_ratio))
    
    n = len(x)
    n_trim = int(n * trim_ratio)
    
    # If trimming would remove all values, return median
    if n_trim * 2 >= n:
        return float(np.median(x))
    
    # Sort and trim
    x_sorted = np.sort(x)
    trimmed_x = x_sorted[n_trim:n - n_trim]
    
    return float(np.mean(trimmed_x))


def winsorize(x: np.ndarray, limits: Tuple[float, float] = (0.01, 0.01)) -> np.ndarray:
    """
    Winsorize (cap) extreme values to specified quantiles.
    
    Values below the lower quantile are set to the lower quantile value,
    and values above the upper quantile are set to the upper quantile value.
    
    Args:
        x: Input array (1D)
        limits: Tuple of (lower_limit, upper_limit) as quantiles (default: (0.01, 0.01))
                This means cap at 1st and 99th percentiles
    
    Returns:
        Winsorized array (same shape as input)
    """
    if len(x) == 0:
        return x.copy()
    
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    
    # Flatten for processing
    x_flat = x.flatten()
    
    # Remove NaNs for quantile computation
    valid_mask = ~np.isnan(x_flat)
    if not np.any(valid_mask):
        return x.copy()
    
    valid_x = x_flat[valid_mask]
    
    # Compute quantiles
    lower_quantile = np.percentile(valid_x, limits[0] * 100)
    upper_quantile = np.percentile(valid_x, (1 - limits[1]) * 100)
    
    # Cap values
    x_flat = np.where(x_flat < lower_quantile, lower_quantile, x_flat)
    x_flat = np.where(x_flat > upper_quantile, upper_quantile, x_flat)
    
    # Restore original shape
    return x_flat.reshape(original_shape)


def hampel_filter(x: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> Tuple[np.ndarray, int]:
    """
    Apply Hampel filter to replace outliers with rolling median.
    
    The Hampel filter identifies outliers by comparing each point to the rolling
    median and MAD within a window. Points that deviate by more than n_sigmas
    standard deviations (scaled by MAD) are replaced with the rolling median.
    
    Args:
        x: Input array (1D)
        window: Window size for rolling median/MAD (default: 7)
        n_sigmas: Number of standard deviations for outlier threshold (default: 3.0)
    
    Returns:
        Tuple of (filtered_array, n_outliers_replaced)
    """
    if len(x) == 0:
        return x.copy(), 0
    
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.flatten()
    n = len(x_flat)
    
    if n < window:
        # Too short for filtering, return as-is
        return x.copy(), 0
    
    filtered = x_flat.copy()
    n_outliers = 0
    
    # Half window for symmetric windowing
    half_window = window // 2
    
    for i in range(n):
        # Define window bounds (handle edges)
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        window_data = x_flat[start_idx:end_idx]
        window_data = window_data[~np.isnan(window_data)]
        
        if len(window_data) == 0:
            continue
        
        # Compute rolling median and MAD
        window_median = np.median(window_data)
        window_deviations = np.abs(window_data - window_median)
        window_mad_value = np.median(window_deviations)
        
        # Threshold: n_sigmas * MAD (scaled to approximate std)
        # MAD ≈ 0.6745 * std for normal distribution, so we scale by 1.4826
        threshold = n_sigmas * window_mad_value * 1.4826
        
        # Check if current point is an outlier
        if not np.isnan(x_flat[i]):
            deviation = np.abs(x_flat[i] - window_median)
            if deviation > threshold:
                filtered[i] = window_median
                n_outliers += 1
    
    # Restore original shape
    if x.ndim > 1:
        filtered = filtered.reshape(x.shape)
    
    return filtered, n_outliers


def downsample_series(x: np.ndarray, max_points: int = 500, method: str = "uniform") -> np.ndarray:
    """
    Downsample a time series to a maximum number of points.
    
    If the series is longer than max_points, select evenly spaced points.
    This is useful for reducing computational cost on very long series.
    
    Args:
        x: Input array (1D)
        max_points: Maximum number of points to keep (default: 500)
        method: Downsampling method (currently only "uniform" supported)
    
    Returns:
        Downsampled array (1D)
    """
    if len(x) == 0:
        return x.copy()
    
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.flatten()
    n = len(x_flat)
    
    if n <= max_points:
        return x.copy()
    
    if method == "uniform":
        # Select evenly spaced indices
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        return x_flat[indices]
    else:
        raise ValueError(f"Unknown downsampling method: {method}")
