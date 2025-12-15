"""
Visualization utilities for sperm trajectories and metrics.

Creates publication-quality plots for scientific analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import matplotlib.colors as mcolors


def plot_trajectories(
    trajectories: List[np.ndarray],
    pixel_size_um: float = 0.1,
    color_by: str = 'track_id',
    velocities: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: str = 'Sperm Trajectories',
    save_path: Optional[str] = None
):
    """
    Plot multiple trajectories.
    
    Args:
        trajectories: List of trajectory arrays, each shape (n_points, 2).
        pixel_size_um: Micrometers per pixel for axis labeling.
        color_by: 'track_id' or 'velocity'.
        velocities: List of velocities for color coding (if color_by='velocity').
        figsize: Figure size.
        title: Plot title.
        save_path: Path to save figure. If None, displays instead.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set color scheme
    if color_by == 'track_id':
        colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))
    elif color_by == 'velocity' and velocities is not None:
        norm = mcolors.Normalize(vmin=min(velocities), vmax=max(velocities))
        cmap = plt.cm.viridis
        colors = [cmap(norm(v)) for v in velocities]
    else:
        colors = ['blue'] * len(trajectories)
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        if len(traj) == 0:
            continue
        
        traj_um = traj * pixel_size_um
        ax.plot(traj_um[:, 0], traj_um[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)
        
        # Mark start and end
        ax.plot(traj_um[0, 0], traj_um[0, 1], 'o', color=colors[i], markersize=8, label=f'Start {i+1}')
        ax.plot(traj_um[-1, 0], traj_um[-1, 1], 's', color=colors[i], markersize=6)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if color_by == 'velocity' and velocities is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Velocity (μm/s)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_velocity_distributions(
    metrics_list: List[Dict],
    metric_names: List[str] = ['VCL', 'VSL', 'VAP', 'LIN'],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot distributions of velocity metrics.
    
    Args:
        metrics_list: List of metric dictionaries.
        metric_names: Names of metrics to plot.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    n_metrics = len(metric_names)
    nrows = (n_metrics + 1) // 2
    ncols = 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric_name in enumerate(metric_names):
        values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
        
        if len(values) == 0:
            continue
        
        ax = axes[i]
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(f'{metric_name}', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.axvline(np.median(values), color='red', linestyle='--', linewidth=2, label='Median')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_MSD_curve(
    lags: np.ndarray,
    msd_values: np.ndarray,
    fps: float,
    fit_params: Optional[Dict] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot Mean Squared Displacement vs. time lag.
    
    Args:
        lags: Time lags (frames).
        msd_values: MSD values (μm²).
        fps: Frames per second.
        fit_params: Optional dictionary with 'D' and 'alpha' for power-law fit.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time_lags = lags / fps
    
    ax.plot(time_lags, msd_values, 'o-', linewidth=2, markersize=6, label='MSD Data')
    
    # Plot fit if provided
    if fit_params is not None:
        D = fit_params.get('D', 0)
        alpha = fit_params.get('alpha', 1)
        fit_msd = 4 * D * time_lags**alpha
        ax.plot(time_lags, fit_msd, '--', linewidth=2, color='red', 
                label=f'Fit: 4D·t^α (D={D:.2f}, α={alpha:.2f})')
    
    ax.set_xlabel('Time Lag (s)', fontsize=12)
    ax.set_ylabel('MSD (μm²)', fontsize=12)
    ax.set_title('Mean Squared Displacement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_turning_angle_distribution(
    angles: np.ndarray,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
):
    """
    Plot turning angle distribution on polar plot.
    
    Args:
        angles: Array of turning angles in radians.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Create histogram bins
    n_bins = 36
    bins = np.linspace(0, 2*np.pi, n_bins + 1)
    
    # Histogram
    counts, bin_edges = np.histogram(angles, bins=bins)
    
    # Plot
    width = 2 * np.pi / n_bins
    bars = ax.bar(bin_edges[:-1], counts, width=width, alpha=0.7, edgecolor='black')
    
    ax.set_title('Turning Angle Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(
    metrics_list: List[Dict],
    metric_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix of metrics.
    
    Args:
        metrics_list: List of metric dictionaries.
        metric_names: Names of metrics to include. If None, uses all numeric metrics.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if metric_names is not None:
        numeric_df = numeric_df[[m for m in metric_names if m in numeric_df.columns]]
    
    # Compute correlation matrix
    corr = numeric_df.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'shrink': 0.8}, ax=ax)
    
    ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
