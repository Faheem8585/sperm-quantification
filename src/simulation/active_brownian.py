"""
Active Brownian Particle (ABP) model for simulating sperm-like trajectories.

This module implements a stochastic model for self-propelled particles,
capturing the essential physics of sperm swimming dynamics.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ABPParameters:
    """Parameters for Active Brownian Particle model."""
    
    v0: float = 50.0  # Self-propulsion speed (μm/s)
    Dr: float = 0.5  # Rotational diffusion coefficient (rad²/s)
    Dt: float = 1.0  # Translational diffusion coefficient (μm²/s)
    dt: float = 0.033  # Time step (s)
    
    # Domain boundaries
    width: float = 500.0  # μm
    height: float = 500.0  # μm
    boundary: str = "reflective"  # "reflective" or "periodic"


class ActiveBrownianParticle:
    """
    Single Active Brownian Particle simulator.
    
    Implements the overdamped Langevin equations:
        dx/dt = v₀ cos(θ) + √(2Dₜ) ηₓ
        dy/dt = v₀ sin(θ) + √(2Dₜ) ηᵧ
        dθ/dt = √(2Dᵣ) ηθ
    
    where η are Gaussian white noise terms.
    """
    
    def __init__(self, params: ABPParameters, initial_pos: Optional[Tuple[float, float]] = None):
        """
        Initialize ABP.
        
        Args:
            params: ABP parameters.
            initial_pos: Initial (x, y) position. If None, random within domain.
        """
        self.params = params
        
        # Initialize position
        if initial_pos is None:
            self.x = np.random.uniform(0, params.width)
            self.y = np.random.uniform(0, params.height)
        else:
            self.x, self.y = initial_pos
        
        # Initialize orientation (random)
        self.theta = np.random.uniform(0, 2 * np.pi)
        
        # Trajectory storage
        self.trajectory = [(self.x, self.y, self.theta)]
    
    def step(self) -> Tuple[float, float, float]:
        """
        Perform one time step of the simulation.
        
        Returns:
            Tuple of (x, y, theta) after the step.
        """
        dt = self.params.dt
        
        # Translational noise
        noise_x = np.sqrt(2 * self.params.Dt * dt) * np.random.randn()
        noise_y = np.sqrt(2 * self.params.Dt * dt) * np.random.randn()
        
        # Update position
        self.x += self.params.v0 * np.cos(self.theta) * dt + noise_x
        self.y += self.params.v0 * np.sin(self.theta) * dt + noise_y
        
        # Apply boundary conditions
        self._apply_boundaries()
        
        # Rotational noise
        noise_theta = np.sqrt(2 * self.params.Dr * dt) * np.random.randn()
        
        # Update orientation
        self.theta += noise_theta
        self.theta = np.mod(self.theta, 2 * np.pi)  # Keep in [0, 2π]
        
        # Store trajectory
        self.trajectory.append((self.x, self.y, self.theta))
        
        return self.x, self.y, self.theta
    
    def _apply_boundaries(self):
        """Apply boundary conditions to particle position."""
        if self.params.boundary == "reflective":
            # Reflect at walls
            if self.x < 0:
                self.x = -self.x
                self.theta = np.pi - self.theta
            elif self.x > self.params.width:
                self.x = 2 * self.params.width - self.x
                self.theta = np.pi - self.theta
            
            if self.y < 0:
                self.y = -self.y
                self.theta = -self.theta
            elif self.y > self.params.height:
                self.y = 2 * self.params.height - self.y
                self.theta = -self.theta
        
        elif self.params.boundary == "periodic":
            # Periodic boundaries
            self.x = np.mod(self.x, self.params.width)
            self.y = np.mod(self.y, self.params.height)
    
    def simulate(self, duration: float) -> np.ndarray:
        """
        Simulate trajectory for specified duration.
        
        Args:
            duration: Simulation duration (seconds).
        
        Returns:
            Array of shape (n_steps, 3) containing (x, y, theta) at each time.
        """
        n_steps = int(duration / self.params.dt)
        
        for _ in range(n_steps):
            self.step()
        
        return np.array(self.trajectory)
    
    def get_trajectory_xy(self) -> np.ndarray:
        """
        Get (x, y) trajectory only.
        
        Returns:
            Array of shape (n_steps, 2) containing (x, y).
        """
        return np.array([(x, y) for x, y, _ in self.trajectory])


class MultiParticleABP:
    """Simulate multiple Active Brownian Particles simultaneously."""
    
    def __init__(self, n_particles: int, params: ABPParameters):
        """
        Initialize multiple particles.
        
        Args:
            n_particles: Number of particles.
            params: ABP parameters (same for all particles).
        """
        self.n_particles = n_particles
        self.params = params
        self.particles = [ActiveBrownianParticle(params) for _ in range(n_particles)]
    
    def simulate(self, duration: float) -> List[np.ndarray]:
        """
        Simulate all particles.
        
        Args:
            duration: Simulation duration (seconds).
        
        Returns:
            List of trajectory arrays, one per particle.
        """
        trajectories = []
        
        for particle in self.particles:
            trajectory = particle.simulate(duration)
            trajectories.append(trajectory)
        
        return trajectories
    
    def get_all_positions_at_time(self, time_idx: int) -> np.ndarray:
        """
        Get positions of all particles at a specific time index.
        
        Args:
            time_idx: Time index.
        
        Returns:
            Array of shape (n_particles, 2) with (x, y) positions.
        """
        positions = []
        for particle in self.particles:
            if time_idx < len(particle.trajectory):
                x, y, _ = particle.trajectory[time_idx]
                positions.append([x, y])
        
        return np.array(positions)


class HeterogeneousABP(MultiParticleABP):
    """
    Simulate heterogeneous population with different parameters.
    
    Useful for modeling X vs Y sperm with different swimming characteristics.
    """
    
    def __init__(
        self,
        n_particles: int,
        params_list: List[ABPParameters],
        labels: Optional[List[int]] = None
    ):
        """
        Initialize heterogeneous particles.
        
        Args:
            n_particles: Number of particles.
            params_list: List of ABPParameters for each subpopulation.
            labels: Optional labels for each particle (e.g., 0 for X, 1 for Y).
        """
        self.n_particles = n_particles
        self.params_list = params_list
        
        # Assign parameters to particles
        if labels is None:
            # Randomly assign
            labels = np.random.choice(len(params_list), size=n_particles)
        
        self.labels = labels
        self.particles = [
            ActiveBrownianParticle(params_list[label]) 
            for label in labels
        ]
    
    def get_labels(self) -> np.ndarray:
        """Get particle labels (e.g., X=0, Y=1)."""
        return np.array(self.labels)
