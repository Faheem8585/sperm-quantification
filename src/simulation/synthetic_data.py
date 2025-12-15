"""
Generate synthetic microscopy data from simulated trajectories.

This module converts ABP trajectories into realistic-looking microscopy videos
for algorithm validation.
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2
from scipy.ndimage import gaussian_filter

from .active_brownian import MultiParticleABP, ABPParameters


class SyntheticMicroscopyGenerator:
    """
    Generate synthetic microscopy videos from particle trajectories.
    
    Simulates:
    - Sperm-like objects (Gaussian blobs)
    - Photon noise (Poisson)
    - Background noise (Gaussian)
    - Motion blur
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        pixel_size_um: float = 0.1,
        particle_intensity: float = 200,
        background_level: float = 50,
        noise_level: float = 10,
        particle_sigma_pixels: float = 2.0
    ):
        """
        Initialize synthetic microscopy generator.
        
        Args:
            image_size: (height, width) in pixels.
            pixel_size_um: Physical size of one pixel (μm).
            particle_intensity: Peak intensity of particle (0-255).
            background_level: Background intensity level.
            noise_level: Standard deviation of Gaussian noise.
            particle_sigma_pixels: Gaussian sigma for particle appearance.
        """
        self.image_size = image_size
        self.pixel_size_um = pixel_size_um
        self.particle_intensity = particle_intensity
        self.background_level = background_level
        self.noise_level = noise_level
        self.particle_sigma_pixels = particle_sigma_pixels
    
    def _um_to_pixels(self, um_coords: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from micrometers to pixels.
        
        Args:
            um_coords: Array of shape (..., 2) with (x, y) in μm.
        
        Returns:
            Array of shape (..., 2) with (x, y) in pixels.
        """
        return um_coords / self.pixel_size_um
    
    def _render_particle(self, x: float, y: float) -> np.ndarray:
        """
        Render a single particle as a Gaussian blob.
        
        Args:
            x, y: Particle position in pixels.
        
        Returns:
            Image with single particle.
        """
        image = np.zeros(self.image_size, dtype=np.float32)
        
        # Create coordinate grid
        yy, xx = np.ogrid[:self.image_size[0], :self.image_size[1]]
        
        # Gaussian blob
        r2 = (xx - x)**2 + (yy - y)**2
        particle = self.particle_intensity * np.exp(-r2 / (2 * self.particle_sigma_pixels**2))
        
        image += particle
        
        return image
    
    def render_frame(self, positions: np.ndarray) -> np.ndarray:
        """
        Render a single frame with multiple particles.
        
        Args:
            positions: Array of shape (n_particles, 2) with (x, y) in μm.
        
        Returns:
            Rendered frame as uint8 image.
        """
        # Initialize with background
        frame = np.ones(self.image_size, dtype=np.float32) * self.background_level
        
        # Convert positions to pixels
        positions_px = self._um_to_pixels(positions)
        
        # Render each particle
        for x, y in positions_px:
            # Check if particle is within image bounds
            if 0 <= x < self.image_size[1] and 0 <= y < self.image_size[0]:
                particle_img = self._render_particle(x, y)
                frame += particle_img
        
        # Add Poisson noise (photon shot noise)
        frame = np.random.poisson(frame).astype(np.float32)
        
        # Add Gaussian noise
        frame += np.random.normal(0, self.noise_level, self.image_size)
        
        # Clip to valid range
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame
    
    def generate_video_from_trajectories(
        self,
        trajectories: List[np.ndarray],
        fps: int = 30
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate synthetic video from particle trajectories.
        
        Args:
            trajectories: List of trajectory arrays, each of shape (n_steps, 3)
                         with (x, y, theta) in μm and radians.
            fps: Frames per second.
        
        Returns:
            Tuple of:
                - Video as array of shape (n_frames, height, width)
                - Metadata dictionary
        """
        # Get number of frames
        n_frames = min(len(traj) for traj in trajectories)
        
        # Pre-allocate video array
        video = np.zeros((n_frames, *self.image_size), dtype=np.uint8)
        
        # Render each frame
        for t in range(n_frames):
            positions = np.array([traj[t, :2] for traj in trajectories])
            video[t] = self.render_frame(positions)
        
        # Create metadata
        metadata = {
            'n_frames': n_frames,
            'fps': fps,
            'pixel_size_um': self.pixel_size_um,
            'image_size': self.image_size,
            'n_particles': len(trajectories),
            'duration_s': n_frames / fps
        }
        
        return video, metadata
    
    def save_video(
        self,
        video: np.ndarray,
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v'
    ):
        """
        Save video to file.
        
        Args:
            video: Video array of shape (n_frames, height, width).
            output_path: Output file path.
            fps: Frames per second.
            codec: Video codec fourcc code.
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        height, width = video.shape[1:3]
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
        
        for frame in video:
            writer.write(frame)
        
        writer.release()


def generate_synthetic_dataset(
    n_particles: int = 50,
    duration: float = 10.0,
    params: Optional[ABPParameters] = None,
    save_path: Optional[str] = None,
    fps: int = 30
) -> Tuple[np.ndarray, List[np.ndarray], dict]:
    """
    High-level function to generate complete synthetic dataset.
    
    Args:
        n_particles: Number of particles to simulate.
        duration: Simulation duration in seconds.
        params: ABP parameters. If None, uses defaults.
        save_path: Optional path to save video file.
        fps: Frames per second for video.
    
    Returns:
        Tuple of:
            - Video array
            - List of ground truth trajectories
            - Metadata dictionary
    
    Example:
        >>> video, trajectories, metadata = generate_synthetic_dataset(
        ...     n_particles=20, duration=5.0
        ... )
        >>> print(f"Generated {metadata['n_frames']} frames")
    """
    # Use default parameters if not provided
    if params is None:
        params = ABPParameters()
    
    # Simulate particles
    sim = MultiParticleABP(n_particles, params)
    trajectories = sim.simulate(duration)
    
    # Generate video
    generator = SyntheticMicroscopyGenerator()
    video, metadata = generator.generate_video_from_trajectories(trajectories, fps=fps)
    
    # Add ground truth to metadata
    metadata['ground_truth_trajectories'] = trajectories
    metadata['abp_parameters'] = {
        'v0': params.v0,
        'Dr': params.Dr,
        'Dt': params.Dt,
        'dt': params.dt
    }
    
    # Save if requested
    if save_path is not None:
        generator.save_video(video, save_path, fps=fps)
        metadata['video_path'] = save_path
    
    return video, trajectories, metadata
