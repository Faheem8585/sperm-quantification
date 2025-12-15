"""
Multi-object tracker for sperm trajectories.

Implements detection-to-track association with Kalman filtering.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

try:
    from ..detection import Detection
    from .kalman import KalmanFilter
except ImportError:
    from detection.blob_detector import Detection
    from tracking.kalman import KalmanFilter


@dataclass
class Track:
    """Represents a single sperm trajectory."""
    
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    kalman_filter: Optional[KalmanFilter] = None
    age: int = 0  # Frames since track creation
    hits: int = 0  # Number of successful detections
    misses: int = 0  # Consecutive missed detections
    active: bool = True
    
    def update(self, detection: Detection, frame_id: int):
        """
        Update track with new detection.
        
        Args:
            detection: Detection object.
            frame_id: Current frame number.
        """
        self.positions.append((detection.x, detection.y))
        self.frame_ids.append(frame_id)
        self.hits += 1
        self.misses = 0
        
        if self.kalman_filter is not None:
            self.kalman_filter.update((detection.x, detection.y))
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next position using Kalman filter.
        
        Returns:
            Predicted (x, y) position.
        """
        if self.kalman_filter is not None:
            return self.kalman_filter.predict()
        elif len(self.positions) > 0:
            return self.positions[-1]
        else:
            return (0, 0)
    
    def mark_missed(self):
        """Mark that track was not matched in current frame."""
        self.misses += 1
    
    def get_last_position(self) -> Optional[Tuple[float, float]]:
        """Get last known position."""
        if len(self.positions) > 0:
            return self.positions[-1]
        return None
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get full trajectory as numpy array.
        
        Returns:
            Array of shape (n_points, 2) with (x, y) positions.
        """
        return np.array(self.positions)
    

def compute_distance_matrix(
    tracks: List[Track],
    detections: List[Detection]
) -> np.ndarray:
    """
    Compute pairwise distance matrix between track predictions and detections.
    
    Args:
        tracks: List of active tracks.
        detections: List of detections in current frame.
    
    Returns:
        Distance matrix of shape (n_tracks, n_detections).
    """
    n_tracks = len(tracks)
    n_detections = len(detections)
    
    if n_tracks == 0 or n_detections == 0:
        return np.array([]).reshape(n_tracks, n_detections)
    
    dist_matrix = np.zeros((n_tracks, n_detections))
    
    for i, track in enumerate(tracks):
        pred_x, pred_y = track.predict()
        
        for j, detection in enumerate(detections):
            dx = pred_x - detection.x
            dy = pred_y - detection.y
            dist = np.sqrt(dx**2 + dy**2)
            dist_matrix[i, j] = dist
    
    return dist_matrix


class SpermTracker:
    """
    Multi-Object Tracker for sperm trajectories.
    
    Uses Hungarian algorithm for optimal detection-to-track association
    combined with Kalman filtering for motion prediction.
    """
    
    def __init__(
        self,
        max_distance: float = 30.0,
        max_gap: int = 5,
        min_track_length: int = 10,
        use_kalman: bool = True,
        dt: float = 1.0
    ):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum distance for associating detection with track (pixels).
            max_gap: Maximum number of frames a track can be missed before termination.
            min_track_length: Minimum track length to keep (frames).
            use_kalman: Whether to use Kalman filtering.
            dt: Time step for Kalman filter.
        """
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.min_track_length = min_track_length
        self.use_kalman = use_kalman
        self.dt = dt
        
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.frame_count = 0
        
        # Store completed tracks
        self.completed_tracks: List[Track] = []
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with detections from current frame.
        
        Args:
            detections: List of detections in current frame.
        
        Returns:
            List of currently active tracks.
        """
        self.frame_count += 1
        
        # Get active tracks
        active_tracks = [t for t in self.tracks if t.active]
        
        # Associate detections with tracks
        matched_indices, unmatched_tracks, unmatched_detections = self._associate(
            active_tracks, detections
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            track = active_tracks[track_idx]
            detection = detections[det_idx]
            track.update(detection, self.frame_count)
            track.age += 1
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            track.mark_missed()
            track.age += 1
            
            # Terminate tracks with too many misses
            if track.misses > self.max_gap:
                track.active = False
                
                # Save if meets minimum length
                if len(track.positions) >= self.min_track_length:
                    self.completed_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self._create_track(detection)
        
        return [t for t in self.tracks if t.active]
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with tracks using Hungarian algorithm.
        
        Args:
            tracks: List of active tracks.
            detections: List of detections.
        
        Returns:
            Tuple of (matched_indices, unmatched_tracks, unmatched_detections).
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        # Compute distance matrix
        dist_matrix = compute_distance_matrix(tracks, detections)
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Filter assignments based on distance threshold
        matched = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for track_idx, det_idx in zip(row_ind, col_ind):
            if dist_matrix[track_idx, det_idx] <= self.max_distance:
                matched.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
            else:
                unmatched_tracks.append(track_idx)
        
        # Add tracks that weren't matched at all
        matched_track_indices = [m[0] for m in matched]
        for i in range(len(tracks)):
            if i not in matched_track_indices and i not in unmatched_tracks:
                unmatched_tracks.append(i)
        
        return matched, unmatched_tracks, unmatched_detections
    
    def _create_track(self, detection: Detection):
        """
        Create new track from detection.
        
        Args:
            detection: Detection object.
        """
        # Initialize Kalman filter if enabled
        kalman_filter = None
        if self.use_kalman:
            kalman_filter = KalmanFilter(
                initial_position=(detection.x, detection.y),
                dt=self.dt
            )
        
        track = Track(
            track_id=self.next_track_id,
            kalman_filter=kalman_filter
        )
        track.update(detection, self.frame_count)
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def get_all_tracks(self) -> List[Track]:
        """
        Get all tracks (active + completed).
        
        Returns:
            List of all tracks.
        """
        # Finalize remaining active tracks
        for track in self.tracks:
            if track.active and len(track.positions) >= self.min_track_length:
                track.active = False
                self.completed_tracks.append(track)
        
        return self.completed_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID.
        
        Args:
            track_id: Track ID.
        
        Returns:
            Track object or None if not found.
        """
        for track in self.tracks + self.completed_tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def export_trajectories(self) -> Dict[int, np.ndarray]:
        """
        Export all trajectories as dictionary.
        
        Returns:
            Dictionary mapping track_id to trajectory array.
        """
        all_tracks = self.get_all_tracks()
        
        trajectories = {}
        for track in all_tracks:
            trajectories[track.track_id] = track.get_trajectory()
        
        return trajectories
