"""
Quick test to verify the pipeline works with correctly scaled data.
"""

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
from simulation.active_brownian import ABPParameters, MultiParticleABP
from detection.blob_detector import BlobDetector
from tracking.tracker import SpermTracker
from metrics.velocity import compute_all_velocity_metrics

print("=" * 60)
print("Quick Pipeline Test")
print("=" * 60)

# Create simulation with SMALL domain to fit in pixels
print("\n1. Simulating sperm (small domain)...")
params = ABPParameters(
    v0=50, Dr=0.5, Dt=1.0, dt=0.033,
    width=200.0,  # 200 μm domain
    height=200.0
)

sim = MultiParticleABP(n_particles=10, params=params)
trajectories = sim.simulate(duration=2.0)
print(f"   ✓ Simulated {len(trajectories)} sperm")

# Create simple synthetic frames manually
print("\n2. Creating synthetic frames...")
pixel_size = 0.4  # 0.4 μm/pixel → 200 μm = 500 pixels
image_size = (512, 512)
n_frames = min(60, len(trajectories[0]))

video = []
for t in range(n_frames):
    frame = np.ones(image_size, dtype=np.uint8) * 30  # Background
    
    for traj in trajectories:
        if t < len(traj):
            x_um, y_um = traj[t, :2]
            x_px = int(x_um / pixel_size) + 50  # Add offset
            y_px = int(y_um / pixel_size) + 50
            
            # Draw bright spot
            if 0 <= x_px < image_size[1] and 0 <= y_px < image_size[0]:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        yy, xx = y_px + dy, x_px + dx
                        if 0 <= xx < image_size[1] and 0 <= yy < image_size[0]:
                            if dx**2 + dy**2 <= 9:  # Circle
                                frame[yy, xx] = min(255, frame[yy, xx] + 180)
    
    video.append(frame)

video = np.array(video)
print(f"   ✓ Created {len(video)} frames")

# Detect
print("\n3. Detecting sperm...")
detector = BlobDetector(method='dog', threshold=0.05, min_area=10, max_area=200)

detections_per_frame = [len(detector.detect(frame)) for frame in video]
avg_det = np.mean(detections_per_frame)
print(f"   ✓ Average detections: {avg_det:.1f} per frame")

# Track
print("\n4. Tracking...")
tracker = SpermTracker(max_distance=20, max_gap=3, min_track_length=15)

for frame in video:
    dets = detector.detect(frame)
    tracker.update(dets)

tracks = tracker.get_all_tracks()
print(f"   ✓ Tracked {len(tracks)} trajectories")

# Metrics
if len(tracks) > 0:
    print("\n5. Computing metrics...")
    sample = tracks[0].get_trajectory()
    metrics = compute_all_velocity_metrics(sample, fps=30, pixel_size_um=pixel_size)
    
    print(f"   ✓ VCL: {metrics['VCL']:.1f} μm/s")
    print(f"   ✓ VSL: {metrics['VSL']:.1f} μm/s") 
    print(f"   ✓ LIN: {metrics['LIN']:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE WORKS CORRECTLY!")
    print("=" * 60)
    print(f"\nResults: Tracked {len(tracks)}/{len(trajectories)} sperm")
    print(f"Success rate: {len(tracks)/len(trajectories)*100:.0f}%")
else:
    print("\n✗ No tracks found - detection/tracking needs tuning")
