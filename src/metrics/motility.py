"""High-level motility analysis combining velocity and trajectory metrics."""

from typing import Dict
import numpy as np

from .velocity import compute_all_velocity_metrics, classify_motility
from .trajectory import compute_all_trajectory_metrics


def analyze_single_trajectory(
    trajectory: np.ndarray,
    fps: float,
    pixel_size_um: float
) -> Dict[str, any]:
    """
    Perform complete analysis on a single sperm trajectory.
    
    Args:
        trajectory: Array of shape (n_points, 2) with (x, y) positions in pixels.
        fps: Frames per second.
        pixel_size_um: Micrometers per pixel.
    
    Returns:
        Dictionary containing all computed metrics and classification.
    """
    # Velocity metrics
    velocity_metrics = compute_all_velocity_metrics(trajectory, fps, pixel_size_um)
    
    # Trajectory metrics
    trajectory_metrics = compute_all_trajectory_metrics(trajectory, fps, pixel_size_um)
    
    # Motility classification
    motility_class = classify_motility(velocity_metrics)
    
    # Combine all metrics
    results = {
        **velocity_metrics,
        **trajectory_metrics,
        'motility_classification': motility_class,
        'track_length': len(trajectory),
        'track_duration_s': len(trajectory) / fps
    }
    
    return results


__all__ = ['analyze_single_trajectory']
