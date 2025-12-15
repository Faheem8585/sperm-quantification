"""
Velocity metrics computation for sperm motility analysis.

Implements WHO 2021 standardized metrics:
- VCL (Curvilinear Velocity)
- VSL (Straight-Line Velocity)
- VAP (Average Path Velocity)
- LIN (Linearity)
- WOB (Wobble)
- ALH (Amplitude of Lateral Head displacement)
- BCF (Beat Cross Frequency)
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from typing import Dict, Optional


def compute_path_lengths(
    trajectory: np.ndarray,
    smooth_window: Optional[int] = None
) -> tuple:
    """
    Compute curvilinear and straight-line path lengths.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        smooth_window: Window size for smoothing (for VAP). None = no smoothing.
    
    Returns:
        Tuple of (curvilinear_length, straight_line_length, average_path_length).
    """
    if len(trajectory) < 2:
        return 0.0, 0.0, 0.0
    
    # Curvilinear length: sum of segment lengths
    displacements = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(displacements, axis=1)
    curvilinear_length = np.sum(segment_lengths)
    
    # Straight-line length: Euclidean distance from start to end
    straight_line_length = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # Average path length: smoothed trajectory
    if smooth_window is not None and len(trajectory) > smooth_window:
        smoothed = np.column_stack([
            uniform_filter1d(trajectory[:, 0], size=smooth_window, mode='nearest'),
            uniform_filter1d(trajectory[:, 1], size=smooth_window, mode='nearest')
        ])
        smooth_displacements = np.diff(smoothed, axis=0)
        smooth_lengths = np.linalg.norm(smooth_displacements, axis=1)
        average_path_length = np.sum(smooth_lengths)
    else:
        average_path_length = curvilinear_length
    
    return curvilinear_length, straight_line_length, average_path_length


def compute_velocity_metrics(
    trajectory: np.ndarray,
    fps: float,
    pixel_size_um: float,
    smooth_window: Optional[int] = 5
) -> Dict[str, float]:
    """
    Compute WHO standard velocity metrics.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions in pixels.
        fps: Frames per second.
        pixel_size_um: Micrometers per pixel.
        smooth_window: Window for VAP smoothing (frames).
    
    Returns:
        Dictionary with velocity metrics:
            - VCL: Curvilinear velocity (μm/s)
            - VSL: Straight-line velocity (μm/s)
            - VAP: Average path velocity (μm/s)
            - LIN: Linearity (VSL/VCL)
            - WOB: Wobble (VAP/VCL)
            - STR: Straightness (VSL/VAP)
    """
    if len(trajectory) < 2:
        return {
            'VCL': 0.0, 'VSL': 0.0, 'VAP': 0.0,
            'LIN': 0.0, 'WOB': 0.0, 'STR': 0.0
        }
    
    # Convert pixels to micrometers
    trajectory_um = trajectory * pixel_size_um
    
    # Compute path lengths
    curv_length, straight_length, avg_path_length = compute_path_lengths(
        trajectory_um, smooth_window=smooth_window
    )
    
    # Duration in seconds
    duration = (len(trajectory) - 1) / fps
    
    if duration == 0:
        return {
            'VCL': 0.0, 'VSL': 0.0, 'VAP': 0.0,
            'LIN': 0.0, 'WOB': 0.0, 'STR': 0.0
        }
    
    # Velocity metrics
    VCL = curv_length / duration  # Curvilinear velocity
    VSL = straight_length / duration  # Straight-line velocity
    VAP = avg_path_length / duration  # Average path velocity
    
    # Derived metrics
    LIN = VSL / VCL if VCL > 0 else 0.0  # Linearity
    WOB = VAP / VCL if VCL > 0 else 0.0  # Wobble
    STR = VSL / VAP if VAP > 0 else 0.0  # Straightness
    
    return {
        'VCL': float(VCL),
        'VSL': float(VSL),
        'VAP': float(VAP),
        'LIN': float(LIN),
        'WOB': float(WOB),
        'STR': float(STR)
    }


def compute_ALH(
    trajectory: np.ndarray,
    pixel_size_um: float,
    smooth_window: int = 5
) -> float:
    """
    Compute Amplitude of Lateral Head displacement (ALH).
    
    ALH measures the side-to-side swimming motion.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        pixel_size_um: Micrometers per pixel.
        smooth_window: Window for smoothing average path.
    
    Returns:
        ALH in micrometers.
    """
    if len(trajectory) < smooth_window + 2:
        return 0.0
    
    # Smooth trajectory to get average path
    smoothed = np.column_stack([
        uniform_filter1d(trajectory[:, 0], size=smooth_window, mode='nearest'),
        uniform_filter1d(trajectory[:, 1], size=smooth_window, mode='nearest')
    ])
    
    # Compute lateral displacements from smoothed path
    lateral_displacements = []
    
    for i in range(1, len(trajectory) - 1):
        # Vector along smoothed path
        path_vec = smoothed[i+1] - smoothed[i-1]
        path_vec_norm = np.linalg.norm(path_vec)
        
        if path_vec_norm == 0:
            continue
        
        path_unit = path_vec / path_vec_norm
        
        # Vector from smoothed to actual position
        lateral_vec = trajectory[i] - smoothed[i]
        
        # Project onto perpendicular to path
        lateral_dist = abs(np.cross(lateral_vec, path_unit))
        lateral_displacements.append(lateral_dist)
    
    if len(lateral_displacements) == 0:
        return 0.0
    
    # ALH = mean of lateral displacements
    ALH = np.mean(lateral_displacements) * pixel_size_um
    
    return float(ALH)


def compute_BCF(
    trajectory: np.ndarray,
    fps: float,
    pixel_size_um: float
) -> float:
    """
    Compute Beat Cross Frequency (BCF).
    
    BCF is the frequency of flagellar beating, estimated from
    the number of times the trajectory crosses its average path.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        fps: Frames per second.
        pixel_size_um: Micrometers per pixel.
    
    Returns:
        BCF in Hz.
    """
    if len(trajectory) < 10:
        return 0.0
    
    # Smooth trajectory
    smooth_window = min(5, len(trajectory) // 3)
    smoothed = np.column_stack([
        uniform_filter1d(trajectory[:, 0], size=smooth_window, mode='nearest'),
        uniform_filter1d(trajectory[:, 1], size=smooth_window, mode='nearest')
    ])
    
    # Compute lateral deviations
    lateral_deviations = []
    
    for i in range(1, len(trajectory) - 1):
        path_vec = smoothed[i+1] - smoothed[i-1]
        path_norm = np.linalg.norm(path_vec)
        
        if path_norm == 0:
            lateral_deviations.append(0)
            continue
        
        path_unit = path_vec / path_norm
        lateral_vec = trajectory[i] - smoothed[i]
        
        # Signed lateral deviation
        lateral_dev = np.cross(lateral_vec, path_unit)
        lateral_deviations.append(lateral_dev)
    
    if len(lateral_deviations) < 3:
        return 0.0
    
    # Count zero crossings
    lateral_array = np.array(lateral_deviations)
    zero_crossings = np.sum(np.diff(np.sign(lateral_array)) != 0)
    
    # Duration
    duration = len(trajectory) / fps
    
    # BCF = crossings per second / 2 (each beat = 2 crossings)
    BCF = (zero_crossings / duration) / 2.0
    
    return float(BCF)


def compute_all_velocity_metrics(
    trajectory: np.ndarray,
    fps: float,
    pixel_size_um: float,
    smooth_window: int = 5
) -> Dict[str, float]:
    """
    Compute all velocity and beat metrics.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions.
        fps: Frames per second.
        pixel_size_um: Micrometers per pixel.
        smooth_window: Smoothing window size.
    
    Returns:
        Dictionary with all metrics.
    """
    metrics = compute_velocity_metrics(trajectory, fps, pixel_size_um, smooth_window)
    
    metrics['ALH'] = compute_ALH(trajectory, pixel_size_um, smooth_window)
    metrics['BCF'] = compute_BCF(trajectory, fps, pixel_size_um)
    
    return metrics


def classify_motility(metrics: Dict[str, float]) -> str:
    """
    Classify sperm motility based on WHO criteria.
    
    Args:
        metrics: Dictionary with velocity metrics.
    
    Returns:
        Motility classification:
            - 'progressive': Progressive motility
            - 'non_progressive': Non-progressive motility
            - 'immotile': Immotile
            - 'hyperactivated': Hyperactivated (capacitated)
    """
    VCL = metrics.get('VCL', 0)
    VSL = metrics.get('VSL', 0)
    LIN = metrics.get('LIN', 0)
    ALH = metrics.get('ALH', 0)
    
    # Hyperactivated: high VCL, low LIN, high ALH
    if VCL > 150 and LIN < 0.3 and ALH > 5:
        return 'hyperactivated'
    
    # Progressive: VSL > 25 μm/s and LIN > 0.5
    elif VSL > 25 and LIN > 0.5:
        return 'progressive'
    
    # Non-progressive: some movement but not progressive
    elif VCL > 5:
        return 'non_progressive'
    
    # Immotile
    else:
        return 'immotile'
