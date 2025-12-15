"""
Test tracking accuracy using synthetic data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from simulation import ABPParameters, generate_synthetic_dataset
from detection import BlobDetector
from tracking import SpermTracker


def test_synthetic_tracking():
    """Test tracking on synthetic data with known ground truth."""
    
    print("=" * 60)
    print("Testing Sperm Tracking Pipeline")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    params = ABPParameters(v0=50, Dr=0.5, Dt=1.0)
    
    video, gt_trajectories, metadata = generate_synthetic_dataset(
        n_particles=10,
        duration=3.0,
        params=params,
        fps=30
    )
    
    print(f"   ✓ Generated {metadata['n_frames']} frames with {metadata['n_particles']} sperm")
    
    # Track sperm
    print("\n2. Detecting and tracking...")
    detector = BlobDetector(method='dog', threshold=0.1)
    tracker = SpermTracker(max_distance=30, min_track_length=10)
    
    for frame in video:
        detections = detector.detect(frame)
        tracker.update(detections)
    
    all_tracks = tracker.get_all_tracks()
    print(f"   ✓ Tracked {len(all_tracks)} trajectories")
    
    # Compute metrics
    print("\n3. Computing metrics...")
    from metrics import analyze_single_trajectory
    
    metrics_list = []
    for track in all_tracks:
        trajectory = track.get_trajectory()
        metrics = analyze_single_trajectory(
            trajectory,
            fps=metadata['fps'],
            pixel_size_um=metadata['pixel_size_um']
        )
        metrics_list.append(metrics)
    
    # Summary
    if len(metrics_list) > 0:
        avg_vcl = np.mean([m['VCL'] for m in metrics_list])
        avg_lin = np.mean([m['LIN'] for m in metrics_list])
        
        print(f"   ✓ Average VCL: {avg_vcl:.2f} μm/s")
        print(f"   ✓ Average LIN: {avg_lin:.2f}")
    
    # Tracking accuracy
    accuracy = len(all_tracks) / len(gt_trajectories) * 100
    print(f"\n4. Tracking Accuracy: {accuracy:.1f}%")
    
    if accuracy > 80:
        print("   ✓ PASS: Tracking accuracy > 80%")
        return True
    else:
        print("   ✗ FAIL: Tracking accuracy < 80%")
        return False


if __name__ == '__main__':
    success = test_synthetic_tracking()
    
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED ✓")
    else:
        print("TEST FAILED ✗")
    print("=" * 60)
