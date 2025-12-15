"""
Trajectory-based physics metrics for sperm dynamics.

Computes Mean Squared Displacement (MSD), persistence length,
turning angles, and other physics-based descriptors.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional


def compute_MSD(
    trajectory: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Mean Squared Displacement (MSD) vs. time lag.
    
    MSD(τ) = <|r(t+τ) - r(t)|²>
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        max_lag: Maximum time lag. If None, uses n_points // 4.
    
    Returns:
        Tuple of (lags, msd_values).
        lags: Array of time lags.
        msd_values: Array of MSD values.
    """
    n_points = len(trajectory)
    
    if max_lag is None:
        max_lag = n_points // 4
    
    max_lag = min(max_lag, n_points - 1)
    
    lags = np.arange(1, max_lag + 1)
    msd_values = np.zeros(max_lag)
    
    for lag in lags:
        # Compute squared displacements for this lag
        displacements = trajectory[lag:] - trajectory[:-lag]
        squared_displacements = np.sum(displacements**2, axis=1)
        msd_values[lag - 1] = np.mean(squared_displacements)
    
    return lags, msd_values


def fit_MSD_diffusion(
    lags: np.ndarray,
    msd_values: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    Fit MSD to power-law model: MSD(τ) = 4D τ^α
    
    α = 1: normal diffusion
    α < 1: subdiffusion
    α > 1: superdiffusion (ballistic if α = 2)
    
    Args:
        lags: Time lags (frames).
        msd_values: MSD values.
        dt: Time step (seconds per frame).
    
    Returns:
        Dictionary with:
            - D: Diffusion coefficient (μm²/s)
            - alpha: Anomalous exponent
            - regime: 'ballistic', 'superdiffusive', 'diffusive', or 'subdiffusive'
    """
    # Convert lags to time
    time_lags = lags * dt
    
    # Fit power law: MSD = 4D * t^alpha
    def power_law(t, D, alpha):
        return 4 * D * t**alpha
    
    try:
        params, _ = curve_fit(
            power_law,
            time_lags,
            msd_values,
            p0=[10, 1.0],  # Initial guess
            bounds=([0, 0], [np.inf, 3])  # D > 0, 0 < alpha < 3
        )
        D, alpha = params
    except:
        # Fallback: linear fit for diffusion coefficient
        if len(time_lags) > 1:
            slope = (msd_values[-1] - msd_values[0]) / (time_lags[-1] - time_lags[0])
            D = slope / 4
            alpha = 1.0
        else:
            D, alpha = 0.0, 1.0
    
    # Classify regime
    if alpha > 1.8:
        regime = 'ballistic'
    elif alpha > 1.2:
        regime = 'superdiffusive'
    elif alpha > 0.8:
        regime = 'diffusive'
    else:
        regime = 'subdiffusive'
    
    return {
        'D': float(D),
        'alpha': float(alpha),
        'regime': regime
    }


def compute_velocity_autocorrelation(
    trajectory: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity autocorrelation function.
    
    C(τ) = <v(t) · v(t+τ)> / <v(t) · v(t)>
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        max_lag: Maximum time lag. If None, uses n_points // 4.
    
    Returns:
        Tuple of (lags, autocorr_values).
    """
    # Compute velocities
    velocities = np.diff(trajectory, axis=0)
    n_vel = len(velocities)
    
    if max_lag is None:
        max_lag = n_vel // 4
    
    max_lag = min(max_lag, n_vel - 1)
    
    lags = np.arange(0, max_lag + 1)
    autocorr = np.zeros(max_lag + 1)
    
    # Normalization (variance of velocity)
    v_variance = np.mean(np.sum(velocities**2, axis=1))
    
    if v_variance == 0:
        return lags, autocorr
    
    for lag in lags:
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            # Dot product of velocities separated by lag
            v_products = np.sum(velocities[:-lag] * velocities[lag:], axis=1)
            autocorr[lag] = np.mean(v_products) / v_variance
    
    return lags, autocorr


def compute_persistence_length(
    lags: np.ndarray,
    autocorr: np.ndarray,
    pixel_size_um: float,
    fps: float
) -> float:
    """
    Estimate persistence length from velocity autocorrelation.
    
    Fits exponential decay: C(τ) = exp(-τ/τ_p)
    Persistence length = v₀ * τ_p
    
    Args:
        lags: Time lags (frames).
        autocorr: Autocorrelation values.
        pixel_size_um: Micrometers per pixel.
        fps: Frames per second.
    
    Returns:
        Persistence length in micrometers.
    """
    # Convert lags to time
    time_lags = lags / fps
    
    # Filter positive autocorrelation values
    positive_idx = autocorr > 0
    if np.sum(positive_idx) < 2:
        return 0.0
    
    time_filt = time_lags[positive_idx]
    autocorr_filt = autocorr[positive_idx]
    
    # Fit exponential decay
    def exp_decay(t, tau_p):
        return np.exp(-t / tau_p)
    
    try:
        params, _ = curve_fit(
            exp_decay,
            time_filt,
            autocorr_filt,
            p0=[1.0],
            bounds=([0], [np.inf])
        )
        tau_p = params[0]
    except:
        # Fallback: find where autocorr drops to 1/e
        try:
            idx = np.where(autocorr_filt < 1/np.e)[0][0]
            tau_p = time_filt[idx]
        except:
            tau_p = 0.0
    
    # Estimate characteristic velocity from trajectory
    if len(time_filt) > 1:
        v0 = pixel_size_um * fps  # Rough estimate
        persistence_length = v0 * tau_p
    else:
        persistence_length = 0.0
    
    return float(persistence_length)


def compute_turning_angles(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute turning angles between consecutive trajectory segments.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
    
    Returns:
        Array of turning angles in radians.
    """
    if len(trajectory) < 3:
        return np.array([])
    
    # Compute displacement vectors
    displacements = np.diff(trajectory, axis=0)
    
    # Compute angles between consecutive displacements
    angles = []
    
    for i in range(len(displacements) - 1):
        v1 = displacements[i]
        v2 = displacements[i + 1]
        
        # Compute angle using dot product
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            continue
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
        
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return np.array(angles)


def compute_directional_persistence(trajectory: np.ndarray) -> float:
    """
    Compute directional persistence as correlation of direction over time.
    
    Returns value in [0, 1] where:
    - 1 = perfectly straight motion
    - 0 = completely random direction
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
    
    Returns:
        Directional persistence score.
    """
    angles = compute_turning_angles(trajectory)
    
    if len(angles) == 0:
        return 0.0
    
    # Mean cosine of turning angle
    mean_cos = np.mean(np.cos(angles))
    
    # Map from [-1, 1] to [0, 1]
    persistence = (mean_cos + 1) / 2
    
    return float(persistence)


def compute_all_trajectory_metrics(
    trajectory: np.ndarray,
    fps: float,
    pixel_size_um: float,
    max_lag: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all trajectory-based metrics.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        fps: Frames per second.
        pixel_size_um: Micrometers per pixel.
        max_lag: Maximum lag for MSD computation.
    
    Returns:
        Dictionary with all metrics.
    """
    # Convert to micrometers
    trajectory_um = trajectory * pixel_size_um
    
    # MSD and diffusion
    lags, msd_values = compute_MSD(trajectory_um, max_lag=max_lag)
    dt = 1.0 / fps
    
    if len(lags) > 0:
        diffusion_metrics = fit_MSD_diffusion(lags, msd_values, dt)
    else:
        diffusion_metrics = {'D': 0.0, 'alpha': 1.0, 'regime': 'unknown'}
    
    # Velocity autocorrelation
    vel_lags, vel_autocorr = compute_velocity_autocorrelation(trajectory)
    
    if len(vel_lags) > 1:
        persistence_length = compute_persistence_length(
            vel_lags, vel_autocorr, pixel_size_um, fps
        )
    else:
        persistence_length = 0.0
    
    # Turning angles
    angles = compute_turning_angles(trajectory)
    
    if len(angles) > 0:
        mean_turning_angle = float(np.mean(angles))
        std_turning_angle = float(np.std(angles))
    else:
        mean_turning_angle = 0.0
        std_turning_angle = 0.0
    
    # Directional persistence
    dir_persistence = compute_directional_persistence(trajectory)
    
    return {
        'diffusion_coefficient': diffusion_metrics['D'],
        'anomalous_exponent': diffusion_metrics['alpha'],
        'diffusion_regime': diffusion_metrics['regime'],
        'persistence_length': persistence_length,
        'mean_turning_angle_rad': mean_turning_angle,
        'std_turning_angle_rad': std_turning_angle,
        'directional_persistence': dir_persistence
    }
