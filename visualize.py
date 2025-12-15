#!/usr/bin/env python3
"""
Generate visualizations from the sperm quantification pipeline.
Creates plots showing trajectories, metrics, and physics analysis.
"""

import sys
from pathlib import Path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving

print("\n" + "="*70)
print("  GENERATING VISUALIZATIONS")
print("="*70 + "\n")

# Import modules
from simulation.active_brownian import ABPParameters, MultiParticleABP
from detection.blob_detector import BlobDetector
from tracking.tracker import SpermTracker
from metrics.velocity import compute_all_velocity_metrics
from metrics.trajectory import compute_MSD, fit_MSD_diffusion
from visualization.plotting import plot_trajectories, plot_velocity_distributions

# Create output directory
output_dir = Path('data/results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("🧬 Step 1: Simulating sperm...")
params = ABPParameters(v0=50.0, Dr=0.5, Dt=1.0, dt=0.033, width=200.0, height=200.0)
n_sperm = 15
sim = MultiParticleABP(n_particles=n_sperm, params=params)
trajectories = sim.simulate(duration=3.0)
print(f"   ✓ Simulated {n_sperm} sperm\n")

print("🎥 Step 2: Creating video...")
pixel_size = 0.4
image_size = (512, 512)
n_frames = min(90, len(trajectories[0]))

video = []
for t in range(n_frames):
    frame = np.ones(image_size, dtype=np.uint8) * 30
    for traj in trajectories:
        x_um, y_um = traj[t, :2]
        x_px = int(x_um / pixel_size) + 50
        y_px = int(y_um / pixel_size) + 50
        if 0 <= x_px < image_size[1] and 0 <= y_px < image_size[0]:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = y_px + dy, x_px + dx
                    if 0 <= xx < image_size[1] and 0 <= yy < image_size[0]:
                        if dx**2 + dy**2 <= 9:
                            frame[yy, xx] = min(255, int(frame[yy, xx]) + 180)
    video.append(frame)

video = np.array(video)
print(f"   ✓ Created {len(video)} frames\n")

print("🔍 Step 3: Detecting and tracking...")
detector = BlobDetector(method='dog', threshold=0.05, min_area=10, max_area=200)
tracker = SpermTracker(max_distance=20, max_gap=3, min_track_length=25)

for frame in video:
    dets = detector.detect(frame)
    tracker.update(dets)

tracks = tracker.get_all_tracks()
print(f"   ✓ Tracked {len(tracks)} trajectories\n")

print("📊 Step 4: Computing metrics...")
metrics_list = []
for track in tracks:
    traj = track.get_trajectory()
    metrics = compute_all_velocity_metrics(traj, fps=30, pixel_size_um=pixel_size)
    metrics_list.append(metrics)
print(f"   ✓ Computed metrics for {len(metrics_list)} tracks\n")

# ============================================================================
# VISUALIZATION 1: Sample Video Frames
# ============================================================================
print("📸 Creating Figure 1: Sample video frames...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Synthetic Microscopy Video - Sample Frames', fontsize=16, fontweight='bold')

frame_indices = [0, 15, 30, 45, 60, 75]
for idx, (ax, frame_idx) in enumerate(zip(axes.flat, frame_indices)):
    if frame_idx < len(video):
        ax.imshow(video[frame_idx], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Frame {frame_idx} ({frame_idx/30:.1f}s)', fontsize=12)
        ax.axis('off')

plt.tight_layout()
save_path = output_dir / '01_sample_frames.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {save_path}\n")

# ============================================================================
# VISUALIZATION 2: Tracked Trajectories
# ============================================================================
print("🎯 Creating Figure 2: Tracked trajectories...")
fig, ax = plt.subplots(figsize=(12, 12))

tracked_trajectories = [track.get_trajectory() for track in tracks]
colors = plt.cm.tab20(np.linspace(0, 1, len(tracked_trajectories)))

for i, traj in enumerate(tracked_trajectories):
    traj_um = traj * pixel_size
    ax.plot(traj_um[:, 0], traj_um[:, 1], '-', color=colors[i], alpha=0.7, linewidth=2)
    # Mark start (circle) and end (square)
    ax.plot(traj_um[0, 0], traj_um[0, 1], 'o', color=colors[i], markersize=10)
    ax.plot(traj_um[-1, 0], traj_um[-1, 1], 's', color=colors[i], markersize=8)

ax.set_xlabel('X Position (μm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Position (μm)', fontsize=14, fontweight='bold')
ax.set_title(f'Tracked Sperm Trajectories (n={len(tracks)})', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add legend
from matplotlib.patches import Circle, Rectangle
legend_elements = [
    Circle((0, 0), 1, color='gray', label='Start position'),
    Rectangle((0, 0), 1, 1, color='gray', label='End position')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
save_path = output_dir / '02_trajectories.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {save_path}\n")

# ============================================================================
# VISUALIZATION 3: Velocity Distributions
# ============================================================================
print("📊 Creating Figure 3: Velocity metric distributions...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('WHO-Standardized Motility Metrics', fontsize=16, fontweight='bold')

metric_names = ['VCL', 'VSL', 'VAP', 'LIN', 'WOB', 'ALH']
units = ['(μm/s)', '(μm/s)', '(μm/s)', '', '', '(μm)']

for idx, (ax, metric_name, unit) in enumerate(zip(axes.flat, metric_names, units)):
    values = [m[metric_name] for m in metrics_list]
    
    ax.hist(values, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(values), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
    ax.axvline(np.mean(values), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
    
    ax.set_xlabel(f'{metric_name} {unit}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = output_dir / '03_velocity_distributions.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {save_path}\n")

# ============================================================================
# VISUALIZATION 4: Mean Squared Displacement
# ============================================================================
print("⚛️  Creating Figure 4: Mean Squared Displacement (MSD)...")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot MSD for multiple tracks
for i, track in enumerate(tracks[:5]):  # First 5 tracks
    traj_um = track.get_trajectory() * pixel_size
    lags, msd = compute_MSD(traj_um, max_lag=40)
    
    if len(lags) > 3:
        time_lags = lags / 30  # Convert to seconds
        ax.plot(time_lags, msd, 'o-', alpha=0.6, label=f'Track {i+1}')

# Fit and plot power law for first track
if len(tracks) > 0:
    traj_um = tracks[0].get_trajectory() * pixel_size
    lags, msd = compute_MSD(traj_um, max_lag=40)
    time_lags = lags / 30
    
    if len(lags) > 5:
        fit_params = fit_MSD_diffusion(lags, msd, dt=1.0/30)
        fit_msd = 4 * fit_params['D'] * time_lags**fit_params['alpha']
        ax.plot(time_lags, fit_msd, 'r--', linewidth=3, 
                label=f"Power-law fit: 4D·t^α\nD={fit_params['D']:.1f} μm²/s, α={fit_params['alpha']:.2f}")

ax.set_xlabel('Time Lag (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('MSD (μm²)', fontsize=14, fontweight='bold')
ax.set_title('Mean Squared Displacement Analysis', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
save_path = output_dir / '04_msd_analysis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {save_path}\n")

# ============================================================================
# VISUALIZATION 5: Summary Statistics
# ============================================================================
print("📈 Creating Figure 5: Summary statistics...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pipeline Performance Summary', fontsize=16, fontweight='bold')

# VCL vs VSL scatter
vcl = [m['VCL'] for m in metrics_list]
vsl = [m['VSL'] for m in metrics_list]
ax1.scatter(vcl, vsl, s=100, alpha=0.6, edgecolors='black')
ax1.plot([0, max(vcl)+10], [0, max(vcl)+10], 'r--', label='VSL = VCL')
ax1.set_xlabel('VCL (μm/s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('VSL (μm/s)', fontsize=12, fontweight='bold')
ax1.set_title('Curvilinear vs Straight-Line Velocity', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Track length distribution
track_lengths = [len(t.positions) for t in tracks]
ax2.hist(track_lengths, bins=15, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(np.mean(track_lengths), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(track_lengths):.0f} frames')
ax2.set_xlabel('Track Length (frames)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Track Duration Distribution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Linearity distribution
lin = [m['LIN'] for m in metrics_list]
ax3.hist(lin, bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Progressive threshold (0.5)')
ax3.set_xlabel('Linearity (LIN)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Linearity Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Validation: Expected vs Measured VCL
expected_vcl = params.v0
measured_vcl = np.mean(vcl)
error = abs(measured_vcl - expected_vcl) / expected_vcl * 100

categories = ['Expected\n(Simulation)', 'Measured\n(Pipeline)']
values = [expected_vcl, measured_vcl]
colors_bar = ['lightblue', 'lightcoral']

bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=2, alpha=0.7)
ax4.set_ylabel('VCL (μm/s)', fontsize=12, fontweight='bold')
ax4.set_title(f'Validation: VCL Measurement\nError: {error:.1f}%', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
save_path = output_dir / '05_summary_statistics.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {save_path}\n")

# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("  ✨ VISUALIZATIONS COMPLETE ✨")
print("="*70)
print(f"\n📁 All plots saved to: {output_dir}/")
print(f"\nGenerated {5} visualization files:")
print(f"   1. Sample video frames")
print(f"   2. Tracked trajectories")
print(f"   3. Velocity distributions (WHO metrics)")
print(f"   4. MSD analysis (physics)")
print(f"   5. Summary statistics & validation")
print(f"\nOpen the PNG files to view the results!")
print("="*70 + "\n")
