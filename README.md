# Sperm Quantification and Motility Analysis

**A research-grade Python pipeline for analyzing sperm dynamics from videomicroscopy**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## About This Project

I developed this computational framework during my research into reproductive biophysics and microfluidic technologies. The pipeline combines WHO-standardized clinical metrics with physics-based trajectory analysis to provide comprehensive quantification of sperm swimming dynamics.

### Key Features

✅ **Advanced Tracking**: Multi-object tracking using Kalman filtering and Hungarian algorithm  
✅ **Clinical Standards**: Full implementation of WHO 2021 motility parameters (VCL, VSL, VAP, LIN, WOB, ALH, BCF)  
✅ **Physics-Based Analysis**: Mean squared displacement, persistence length, diffusion coefficients  
✅ **Validation Tools**: Active Brownian particle simulator for ground-truth testing  
✅ **Statistical Analysis**: Population comparison and hypothesis testing capabilities  
✅ **Research-Ready Outputs**: Publication-quality visualizations and data export  

---

## Motivation

My fascination with biophysics led me to explore the intersection of reproductive biology and microfluidic engineering. Recent studies on label-free sperm separation through microfluidic devices (Sexing by Self-Propulsion) inspired me to create this analytical framework.

### What This Pipeline Does

- **Quantifies swimming patterns** with research-grade precision
- **Supports experimental validation** of microfluidic separation techniques  
- **Enables physics-based modeling** of active particles in confined geometries
- **Provides clinical assessment** following WHO 2021 guidelines

### Applications

- **Reproductive Biology**: Motility characterization across species and conditions
- **Microfluidic Engineering**: Device validation and optimization  
- **Clinical Settings**: Sperm quality assessment for assisted reproduction  
- **Fundamental Physics**: Active matter and self-propelled particle dynamics

---

## Installation

### Requirements

- Python 3.9 or higher
- OpenCV, scikit-image, NumPy, SciPy
- Matplotlib, Seaborn for visualization
- FilterPy for Kalman filtering

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/sperm-quantification.git
cd sperm-quantification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Quick Start

### 1. Generate Synthetic Data

```python
from src.simulation import generate_synthetic_dataset

# Generate synthetic sperm trajectories
video, trajectories, metadata = generate_synthetic_dataset(
    n_particles=20,
    duration=5.0,
    save_path="data/synthetic/test_video.avi"
)

print(f"Generated {metadata['n_frames']} frames with {metadata['n_particles']} sperm")
```

### 2. Track Sperm from Video

```python
from src.preprocessing import VideoReader, preprocess_frame
from src.detection import BlobDetector
from src.tracking import SpermTracker

# Initialize components
detector = BlobDetector(method='dog', threshold=0.1)
tracker = SpermTracker(max_distance=30, min_track_length=10)

# Process video
with VideoReader("data/raw/sperm_video.avi") as video:
    for frame in video:
        # Preprocess
        processed = preprocess_frame(frame, denoise_method='bilateral')
        
        # Detect sperm
        detections = detector.detect(processed)
        
        # Update tracker
        active_tracks = tracker.update(detections)

# Get completed trajectories    
all_tracks = tracker.get_all_tracks()
print(f"Tracked {len(all_tracks)} sperm")
```

### 3. Compute Motility Metrics

```python
from src.metrics import analyze_single_trajectory

# Analyze a trajectory
trajectory = all_tracks[0].get_trajectory()  # (n_points, 2) array

metrics = analyze_single_trajectory(
    trajectory,
    fps=30,
    pixel_size_um=0.1
)

print(f"VCL: {metrics['VCL']:.2f} μm/s")
print(f"VSL: {metrics['VSL']:.2f} μm/s")
print(f"LIN: {metrics['LIN']:.2f}")
print(f"Classification: {metrics['motility_classification']}")
```

### 4. Visualize Results

```python
from src.visualization import plot_trajectories, plot_velocity_distributions

# Plot all trajectories
trajectories_list = [track.get_trajectory() for track in all_tracks]
plot_trajectories(trajectories_list, pixel_size_um=0.1, save_path="results/trajectories.png")

# Plot metric distributions
metrics_list = [analyze_single_trajectory(traj, 30, 0.1) for traj in trajectories_list]
plot_velocity_distributions(metrics_list, save_path="results/velocity_dist.png")
```

---

## Pipeline Architecture

```
┌─────────────┐
│ Video Input │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  ← Denoising, background subtraction
└─────┬───────────┘
      │
      ▼
┌──────────────┐
│  Detection   │  ← Blob detection, watershed segmentation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Tracking   │  ← Kalman filter, Hungarian assignment
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Metrics    │  ← VCL, VSL, MSD, persistence
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Visualization│  ← Plots, statistics, export
└──────────────┘
```

---

## Module Documentation

### Preprocessing (`src/preprocessing/`)

- **`video_loader.py`**: Load videos (AVI, MP4, TIFF stacks)
- **`denoising.py`**: Gaussian, bilateral, NLM filtering  
- **`background.py`**: MOG2 and median background subtraction

### Detection (`src/detection/`)

- **`blob_detector.py`**: DoG, LoG, DoH blob detection
- **`segmentation.py`**: Watershed, adaptive thresholding

### Tracking (`src/tracking/`)

- **`kalman.py`**: Kalman filter for motion prediction
- **`tracker.py`**: Multi-object tracker with Hungarian algorithm

### Metrics (`src/metrics/`)

- **`velocity.py`**: VCL, VSL, VAP, LIN, WOB, ALH, BCF (WHO 2021)
- **`trajectory.py`**: MSD, persistence length, turning angles
- **`motility.py`**: High-level analysis combining all metrics

### Simulation (`src/simulation/`)

- **`active_brownian.py`**: Stochastic ABP model
- **`synthetic_data.py`**: Generate realistic microscopy videos

### Visualization (`src/visualization/`)

- **`plotting.py`**: Publication-quality trajectory and metric plots

### Analysis (`src/analysis/`)

- **`statistics.py`**: Population comparison, effect sizes

---

## Configuration

Customize pipeline parameters via YAML files in `configs/`:

```yaml
# configs/default.yaml
detection:
  method: "blob"
  blob:
    threshold: 0.1
    min_area: 10
    max_area: 200

tracking:
  max_distance: 30
  max_gap: 5
  min_track_length: 10

metrics:
  velocity:
    time_window: 1.0
```

Load configuration:

```python
from src.utils import Config

config = Config('configs/microfluidic.yaml')
threshold = config.get('detection.blob.threshold')
```

---

## Example Notebooks

Explore `notebooks/` for detailed tutorials:

1. **`01_synthetic_data_demo.ipynb`**: Generate and visualize synthetic data
2. **`02_basic_analysis.ipynb`**: Complete pipeline walkthrough
3. **`03_advanced_tracking.ipynb`**: Tracking algorithm comparison
4. **`04_statistical_comparison.ipynb`**: X vs Y sperm analysis

---

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Tests cover:
- Tracking accuracy on synthetic data
- Metric computation validation
- ABP parameter recovery

---

## Assumptions and Limitations

### Assumptions

1. **Imaging**: Sufficient resolution to resolve sperm heads (~3-5 μm)
2. **Frame Rate**: Adequate temporal sampling (typically 30-60 Hz)
3. **Contrast**: Sperm appear brighter than background (or vice versa)
4. **Planarity**: Sperm swim primarily in focal plane

### Limitations

1. **3D Motion**: Pipeline analyzes 2D projections; out-of-plane motion not captured
2. **Flagellum**: Optional flagellum detection; focus is on head tracking
3. **Dense Fields**: Performance degrades with > 50 sperm/field
4. **Fixed Parameters**: Some WHO metrics require manual threshold tuning

---

## Connection to SEB Research

This pipeline directly supports **Sexing by Self-Propulsion** research by:

- **Quantifying subtle kinematic differences** between X and Y sperm
- **Validating separation efficiency** in microfluidic devices
- **Providing ground truth** for machine learning classifiers
- **Enabling parameter sweeps** via synthetic data generation

Key metrics for SEB:
- **Linearity (LIN)**: Directional persistence  
- **Beat parameters (ALH, BCF)**: Flagellar dynamics
- **Persistence length**: Swimming strategy characterization

---

## Citation

If you find this pipeline useful in your research, please consider citing:

```bibtex
@software{sperm_quantification_2024,
  author = {Faheem},
  title = {Sperm Quantification and Motility Analysis Pipeline},
  year = {2024},
  url = {https://github.com/Faheem8585/sperm-quantification}
}
```

---

## Contributing

I welcome contributions and suggestions! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project builds upon established methods in computer vision, active matter physics, and clinical semen analysis. I'm grateful to the scientific community for:

- **WHO Laboratory Manual (2021)** - standardized motility criteria
- **Active matter physics research** - theoretical foundations for ABP models
- **OpenCV and scikit-image communities** - excellent computer vision tools

---

**Developed with passion for computational biology and biophysics research**
