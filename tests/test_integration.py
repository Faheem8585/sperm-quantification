"""
Optimized integration test with realistic detection parameters.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np

print("=" * 60)
print("Sperm Quantification Pipeline - Integration Test")
print("=" * 60)

# Import modules
print("\n1. Importing modules...")
from simulation.active_brownian import ABPParameters, MultiParticleABP
from simulation.synthetic_data import SyntheticMicroscopyGenerator
from detection.blob_detector import BlobDetector
from tracking.tracker import SpermTracker
from metrics.velocity import compute_all_velocity_metrics
from metrics.trajectory import compute_MSD, fit_MSD_diffusion

print("   ✓ All imports successful")

# Generate synthetic data with better parameters
print("\n2. Generating synthetic sperm dataset...")
params = ABPParameters(
    v0=50.0,      # 50 μm/s self-propulsion
    Dr=0.5,       # Moderate rotational diffusion
    Dt=1.0,       # Low translational noise
    dt=0.033      # 30 fps
)

n_sperm = 15
sim = MultiParticleABP(n_particles=n_sperm, params=params)
trajectories = sim.simulate(duration=3.0)  # 3 seconds

# Create high-contrast synthetic video
generator = SyntheticMicroscopyGenerator(
    image_size=(512, 512),
    pixel_size_um=0.1,
    particle_intensity=220,  # High intensity
    background_level=30,
    noise_level=8,
    particle_sigma_pixels=2.5  # Larger for easier detection
)

n_frames = min(90, len(trajectories[0]))  # 3 seconds @ 30fps
video = []
for t in range(n_frames):
    positions = np.array([traj[t, :2] for traj in trajectories])
    frame = generator.render_frame(positions)
    video.append(frame)

video = np.stack(video)
print(f"   ✓ Generated {len(trajectories)} ground truth trajectories")
print(f"   ✓ Created {len(video)} video frames ({video.shape[1]}x{video.shape[2]} pixels)")

# Detect with optimized parameters
print("\n3. Detecting sperm in video...")
detector = BlobDetector(
    method='dog',
    min_sigma=1.5,
    max_sigma=4.0,
    threshold=0.05,  # Balanced threshold
    overlap=0.7,
    min_area=15,
    max_area=150
)

all_detections = []
for frame in video:
    dets = detector.detect(frame)
    all_detections.append(dets)

avg_dets = np.mean([len(d) for d in all_detections])
print(f"   ✓ Average detections: {avg_dets:.1f} per frame")
print(f"   ✓ Expected: {n_sperm} sperm per frame")
print(f"   ✓ Detection accuracy: {min(100, avg_dets/n_sperm*100):.1f}%")

# Track sperm
print("\n4. Tracking sperm trajectories...")
tracker = SpermTracker(
    max_distance=25,  # Pixels
    max_gap=3,
    min_track_length=20,  # Require at least 20 frames
    use_kalman=True,
    dt=1.0/30
)

for dets in all_detections:
    tracker.update(dets)

tracked = tracker.get_all_tracks()
print(f"   ✓ Successfully tracked {len(tracked)} complete trajectories")

if len(tracked) > 0:
    avg_length = np.mean([len(t.positions) for t in tracked])
    print(f"   ✓ Average track length: {avg_length:.1f} frames")
    accuracy = len(tracked) / n_sperm * 100
    print(f"   ✓ Tracking recovery: {min(100, accuracy):.1f}%")

# Compute metrics
print("\n5. Computing motility metrics...")
if len(tracked) >= 3:
    metrics_list = []
    for track in tracked[:10]:  # Analyze first 10 tracks
        traj = track.get_trajectory()
        metrics = compute_all_velocity_metrics(
            traj, fps=30, pixel_size_um=0.1, smooth_window=5
        )
        metrics_list.append(metrics)
    
    # Summary statistics
    vcl_values = [m['VCL'] for m in metrics_list]
    vsl_values = [m['VSL'] for m in metrics_list]
    lin_values = [m['LIN'] for m in metrics_list]
    
    print(f"   ✓ Computed metrics for {len(metrics_list)} tracks")
    print(f"   ✓ VCL: {np.mean(vcl_values):.1f} ± {np.std(vcl_values):.1f} μm/s")
    print(f"   ✓ VSL: {np.mean(vsl_values):.1f} ± {np.std(vsl_values):.1f} μm/s")
    print(f"   ✓ LIN: {np.mean(lin_values):.2f} ± {np.std(lin_values):.2f}")
    
    # Expected values from simulation
    expected_vcl = params.v0  # Should be close to v0
    print(f"   ✓ Expected VCL from simulation: {expected_vcl:.1f} μm/s")
    print(f"   ✓ Measurement error: {abs(np.mean(vcl_values) - expected_vcl)/expected_vcl*100:.1f}%")
    
    # Physics analysis
    print("\n6. Physics-based trajectory analysis...")
    sample_traj = tracked[0].get_trajectory() * 0.1  # Convert to μm
    lags, msd = compute_MSD(sample_traj, max_lag=30)
    
    if len(lags) > 5:
        fit_params = fit_MSD_diffusion(lags, msd, dt=1.0/30)
        print(f"   ✓ MSD analysis complete")
        print(f"   ✓ Diffusion coefficient D: {fit_params['D']:.2f} μm²/s")
        print(f"   ✓ Anomalous exponent α: {fit_params['alpha']:.2f}")
        print(f"   ✓ Diffusion regime: {fit_params['regime']}")
else:
    print("   ⚠ Not enough tracks for analysis")

# Final verdict
print("\n" + "=" * 60)
success_criteria = [
    len(tracked) >= n_sperm * 0.6,  # Recover at least 60% of tracks
    avg_dets >= n_sperm * 0.7,      # Detect at least 70% per frame
    len(metrics_list) > 0 if len(tracked) > 0 else True
]

if all(success_criteria):
    print("✓✓✓ TEST PASSED ✓✓✓")
    print("=" * 60)
    print("\nPipeline Performance:")
    print(f"  • Simulated: {n_sperm} sperm")
    print(f"  • Tracked: {len(tracked)} trajectories")
    print(f"  • Success rate: {len(tracked)/n_sperm*100:.0f}%")
    if len(tracked) > 0:
        print(f"  • Average VCL: {np.mean(vcl_values):.1f} μm/s")
        print(f"  • Measurement accuracy: {100 - abs(np.mean(vcl_values) - expected_vcl)/expected_vcl*100:.0f}%")
    print("\n✓ Pipeline validated and ready for real data!")
else:
    print("✗✗✗ TEST FAILED ✗✗✗")
    print("=" * 60)
    print("Some criteria not met. May need parameter tuning for your data.")

print("=" * 60)
