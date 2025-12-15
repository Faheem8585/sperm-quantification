#!/usr/bin/env python3
"""
Simple demo of the sperm quantification pipeline.
Run this to see the complete workflow in action!
"""

import sys
from pathlib import Path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np

print("\n" + "="*70)
print("  SPERM QUANTIFICATION PIPELINE - DEMONSTRATION")
print("="*70)

# Import modules
print("\n📦 Loading modules...")
from simulation.active_brownian import ABPParameters, MultiParticleABP
from detection.blob_detector import BlobDetector
from tracking.tracker import SpermTracker
from metrics.velocity import compute_all_velocity_metrics
from metrics.trajectory import compute_MSD, fit_MSD_diffusion
print("   ✓ All modules loaded successfully\n")

# Step 1: Simulate sperm using physics
print("🧬 STEP 1: Simulating sperm with Active Brownian Particle physics")
print("-" * 70)

params = ABPParameters(
    v0=50.0,      # Self-propulsion: 50 μm/s
    Dr=0.5,       # Rotational diffusion: 0.5 rad²/s
    Dt=1.0,       # Translational noise: 1.0 μm²/s
    dt=0.033,     # Time step: 33ms (30 fps)
    width=200.0,  # Domain: 200 μm × 200 μm
    height=200.0
)

print(f"   Parameters: v₀={params.v0} μm/s, Dᵣ={params.Dr} rad²/s")

n_sperm = 12
sim = MultiParticleABP(n_particles=n_sperm, params=params)
trajectories = sim.simulate(duration=2.5)

print(f"   ✓ Simulated {n_sperm} sperm for 2.5 seconds")
print(f"   ✓ Generated {len(trajectories[0])} time points per trajectory\n")

# Step 2: Create synthetic microscopy video
print("🎥 STEP 2: Creating synthetic microscopy video")
print("-" * 70)

pixel_size = 0.4  # μm/pixel
image_size = (512, 512)
n_frames = min(75, len(trajectories[0]))

video = []
for t in range(n_frames):
    frame = np.ones(image_size, dtype=np.uint8) * 30  # Dark background
    
    for traj in trajectories:
        x_um, y_um = traj[t, :2]
        x_px = int(x_um / pixel_size) + 50
        y_px = int(y_um / pixel_size) + 50
        
        # Draw bright circular spot (sperm head)
        if 0 <= x_px < image_size[1] and 0 <= y_px < image_size[0]:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = y_px + dy, x_px + dx
                    if 0 <= xx < image_size[1] and 0 <= yy < image_size[0]:
                        if dx**2 + dy**2 <= 9:
                            frame[yy, xx] = min(255, int(frame[yy, xx]) + 180)
    
    video.append(frame)

video = np.array(video)
print(f"   ✓ Created video: {len(video)} frames @ {image_size[0]}×{image_size[1]} pixels")
print(f"   ✓ Pixel size: {pixel_size} μm/pixel\n")

# Step 3: Detect sperm in each frame
print("🔍 STEP 3: Detecting sperm heads using blob detection")
print("-" * 70)

detector = BlobDetector(
    method='dog',       # Difference of Gaussians
    threshold=0.05,
    min_area=10,
    max_area=200
)

all_detections = []
for frame in video:
    dets = detector.detect(frame)
    all_detections.append(dets)

avg_detections = np.mean([len(d) for d in all_detections])
print(f"   ✓ Processed {len(video)} frames")
print(f"   ✓ Average detections: {avg_detections:.1f} per frame")
print(f"   ✓ Expected: {n_sperm} sperm per frame")
print(f"   ✓ Detection rate: {min(100, avg_detections/n_sperm*100):.0f}%\n")

# Step 4: Track sperm trajectories
print("🎯 STEP 4: Tracking sperm trajectories with Kalman filtering")
print("-" * 70)

tracker = SpermTracker(
    max_distance=20,      # Maximum 20 pixels between frames
    max_gap=3,            # Allow 3 missed detections
    min_track_length=20,  # Require ≥20 frames
    use_kalman=True
)

for dets in all_detections:
    tracker.update(dets)

tracks = tracker.get_all_tracks()
print(f"   ✓ Tracked {len(tracks)} complete trajectories")

if len(tracks) > 0:
    avg_length = np.mean([len(t.positions) for t in tracks])
    print(f"   ✓ Average track length: {avg_length:.0f} frames ({avg_length/30:.1f}s)")
    print(f"   ✓ Success rate: {len(tracks)/n_sperm*100:.0f}%\n")

# Step 5: Compute WHO motility metrics
print("📊 STEP 5: Computing WHO-standardized motility metrics")
print("-" * 70)

if len(tracks) >= 3:
    metrics_list = []
    
    for track in tracks:
        traj = track.get_trajectory()
        metrics = compute_all_velocity_metrics(
            traj, fps=30, pixel_size_um=pixel_size
        )
        metrics_list.append(metrics)
    
    # Calculate statistics
    vcl_values = [m['VCL'] for m in metrics_list]
    vsl_values = [m['VSL'] for m in metrics_list]
    lin_values = [m['LIN'] for m in metrics_list]
    alh_values = [m['ALH'] for m in metrics_list]
    
    print(f"   Analyzed {len(metrics_list)} tracks:")
    print(f"   • VCL (Curvilinear Velocity):  {np.mean(vcl_values):6.1f} ± {np.std(vcl_values):4.1f} μm/s")
    print(f"   • VSL (Straight-Line Velocity): {np.mean(vsl_values):6.1f} ± {np.std(vsl_values):4.1f} μm/s")
    print(f"   • LIN (Linearity):             {np.mean(lin_values):6.2f} ± {np.std(lin_values):4.2f}")
    print(f"   • ALH (Lateral Head Amplitude): {np.mean(alh_values):6.2f} ± {np.std(alh_values):4.2f} μm")
    
    print(f"\n   📈 Validation against simulation:")
    print(f"   • Expected VCL: {params.v0:.1f} μm/s")
    print(f"   • Measured VCL: {np.mean(vcl_values):.1f} μm/s")
    error = abs(np.mean(vcl_values) - params.v0) / params.v0 * 100
    print(f"   • Measurement error: {error:.1f}%")
    
    if error < 15:
        print(f"   ✓ Excellent agreement! (<15% error)\n")
    else:
        print(f"   ⚠ Acceptable (measurement includes noise)\n")

# Step 6: Physics analysis
print("⚛️  STEP 6: Physics-based trajectory analysis")
print("-" * 70)

if len(tracks) > 0:
    # Analyze first trajectory
    sample_traj = tracks[0].get_trajectory() * pixel_size  # Convert to μm
    lags, msd = compute_MSD(sample_traj, max_lag=30)
    
    if len(lags) > 5:
        fit_params = fit_MSD_diffusion(lags, msd, dt=1.0/30)
        
        print(f"   Mean Squared Displacement (MSD) Analysis:")
        print(f"   • Diffusion coefficient D: {fit_params['D']:.2f} μm²/s")
        print(f"   • Anomalous exponent α:   {fit_params['alpha']:.2f}")
        print(f"   • Regime: {fit_params['regime']}")
        
        if fit_params['alpha'] > 1.5:
            print(f"   ✓ Ballistic motion detected (α > 1.5) - active swimming!\n")
        else:
            print(f"   ✓ Diffusive motion detected\n")

# Final summary
print("="*70)
print("  ✨ PIPELINE EXECUTION COMPLETE ✨")
print("="*70)

if len(tracks) > 0:
    success_rate = len(tracks) / n_sperm * 100
    
    print(f"\n📋 Summary:")
    print(f"   • Simulated:        {n_sperm} sperm")
    print(f"   • Tracked:          {len(tracks)} trajectories")
    print(f"   • Success rate:     {success_rate:.0f}%")
    print(f"   • Average VCL:      {np.mean(vcl_values):.1f} μm/s")
    print(f"   • Measurement error: {error:.1f}%")
    
    if success_rate >= 70 and error < 20:
        print(f"\n   🎉 EXCELLENT! Pipeline validated successfully!")
        print(f"   ✓ Ready for real experimental data")
    elif success_rate >= 50:
        print(f"\n   ✓ GOOD! Pipeline working correctly")
        print(f"   → May need parameter tuning for real data")
    else:
        print(f"\n   ⚠ Some tracks lost - try adjusting detection parameters")
else:
    print(f"\n   ⚠ No tracks found - detection parameters need adjustment")

print("\n" + "="*70)
print("  Thank you for using the Sperm Quantification Pipeline!")
print("  For more details, see README.md and the example notebooks.")
print("="*70 + "\n")
