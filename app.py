#!/usr/bin/env python3
"""
Sperm Quantification Pipeline - Web Interface

A user-friendly Streamlit web app for analyzing sperm motility.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Import pipeline modules
from simulation.active_brownian import ABPParameters, MultiParticleABP
from detection.blob_detector import BlobDetector
from tracking.tracker import SpermTracker
from metrics.velocity import compute_all_velocity_metrics
from metrics.trajectory import compute_MSD, fit_MSD_diffusion

# Page configuration
st.set_page_config(
    page_title="Sperm Quantification Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧬 Sperm Quantification & Motility Analysis</div>', unsafe_allow_html=True)
st.markdown("**A comprehensive pipeline for analyzing sperm dynamics from videomicroscopy**")
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    # Data source
    st.subheader("1. Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Generate Synthetic Data", "Upload Video (Coming Soon)"],
        help="Synthetic data uses physics-based simulation"
    )
    
    # Set default values
    n_sperm = 15
    duration = 2.5
    v0 = 50.0
    Dr = 0.5
    
    if data_source == "Generate Synthetic Data":
        st.subheader("🧬 Simulation Parameters")
        
        n_sperm = st.slider("Number of sperm", 5, 30, 15, help="Number of particles to simulate")
        duration = st.slider("Duration (seconds)", 1.0, 5.0, 2.5, 0.5)
        v0 = st.slider("Self-propulsion speed (μm/s)", 20.0, 100.0, 50.0, 5.0)
        Dr = st.slider("Rotational diffusion (rad²/s)", 0.1, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    
    # Detection parameters
    st.subheader("2. Detection Settings")
    detection_method = st.selectbox("Detection method", ["dog", "log", "doh"], help="Blob detection algorithm")
    detection_threshold = st.slider("Threshold", 0.01, 0.2, 0.05, 0.01, help="Lower = more sensitive")
    min_area = st.slider("Min area (pixels)", 5, 50, 10, 5)
    max_area = st.slider("Max area (pixels)", 50, 500, 200, 50)
    
    st.markdown("---")
    
    # Tracking parameters
    st.subheader("3. Tracking Settings")
    max_distance = st.slider("Max distance (pixels)", 10, 50, 20, 5, help="Max distance between frames")
    max_gap = st.slider("Max gap (frames)", 1, 10, 3, 1, help="Frames to tolerate missing detection")
    min_track_length = st.slider("Min track length (frames)", 10, 50, 25, 5)
    
    st.markdown("---")
    
    # Run button
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content area
if run_analysis:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Generate/Load Data
    status_text.text("🧬 Step 1/5: Generating synthetic data...")
    progress_bar.progress(10)
    
    with st.spinner("Simulating sperm..."):
        params = ABPParameters(
            v0=v0, Dr=Dr, Dt=1.0, dt=0.033,
            width=200.0, height=200.0
        )
        sim = MultiParticleABP(n_particles=n_sperm, params=params)
        trajectories = sim.simulate(duration=duration)
        
        # Create synthetic video
        pixel_size = 0.4
        image_size = (512, 512)
        n_frames = min(int(duration * 30), len(trajectories[0]))
        
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
    
    st.success(f"✓ Generated {n_frames} frames with {n_sperm} sperm")
    progress_bar.progress(20)
    
    # Step 2: Detection
    status_text.text("🔍 Step 2/5: Detecting sperm in each frame...")
    progress_bar.progress(30)
    
    with st.spinner("Running blob detection..."):
        detector = BlobDetector(
            method=detection_method,
            threshold=detection_threshold,
            min_area=min_area,
            max_area=max_area
        )
        
        all_detections = []
        for frame in video:
            dets = detector.detect(frame)
            all_detections.append(dets)
        
        avg_detections = np.mean([len(d) for d in all_detections])
    
    st.success(f"✓ Detected average {avg_detections:.1f} sperm per frame")
    progress_bar.progress(50)
    
    # Step 3: Tracking
    status_text.text("🎯 Step 3/5: Tracking trajectories...")
    progress_bar.progress(60)
    
    with st.spinner("Tracking with Kalman filtering..."):
        tracker = SpermTracker(
            max_distance=max_distance,
            max_gap=max_gap,
            min_track_length=min_track_length,
            use_kalman=True
        )
        
        for dets in all_detections:
            tracker.update(dets)
        
        tracks = tracker.get_all_tracks()
    
    st.success(f"✓ Tracked {len(tracks)} complete trajectories")
    progress_bar.progress(80)
    
    # Step 4: Compute Metrics
    status_text.text("📊 Step 4/5: Computing motility metrics...")
    progress_bar.progress(90)
    
    with st.spinner("Computing WHO metrics..."):
        metrics_list = []
        for track in tracks:
            traj = track.get_trajectory()
            metrics = compute_all_velocity_metrics(traj, fps=30, pixel_size_um=pixel_size)
            metrics['track_id'] = track.track_id
            metrics['track_length'] = len(traj)
            metrics_list.append(metrics)
        
        df_metrics = pd.DataFrame(metrics_list)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    st.markdown("---")
    
    # Results Display
    st.header("📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sperm Tracked", len(tracks), delta=f"{len(tracks)/n_sperm*100:.0f}% of simulated")
    
    with col2:
        avg_vcl = df_metrics['VCL'].mean()
        st.metric("Average VCL", f"{avg_vcl:.1f} μm/s", delta=f"{abs(avg_vcl - v0)/v0*100:.1f}% error")
    
    with col3:
        avg_lin = df_metrics['LIN'].mean()
        st.metric("Average Linearity", f"{avg_lin:.2f}")
    
    with col4:
        progressive = (df_metrics['LIN'] > 0.5).sum()
        st.metric("Progressive", f"{progressive}/{len(tracks)}", delta=f"{progressive/len(tracks)*100:.0f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video & Trajectories", "📊 Metrics", "⚛️ Physics", "💾 Export"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Sample Video Frame")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(video[n_frames//2], cmap='gray')
            ax.set_title(f"Frame {n_frames//2} ({n_frames//2/30:.1f}s)", fontsize=12)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with col_right:
            st.subheader("Tracked Trajectories")
            fig, ax = plt.subplots(figsize=(8, 8))
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(tracks)))
            for i, track in enumerate(tracks):
                traj = track.get_trajectory() * pixel_size
                ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.7, linewidth=2)
                ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=8)
            
            ax.set_xlabel('X Position (μm)', fontsize=12)
            ax.set_ylabel('Y Position (μm)', fontsize=12)
            ax.set_title(f'Tracked Trajectories (n={len(tracks)})', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        st.subheader("WHO-Standardized Motility Metrics")
        
        # Display full metrics table
        display_cols = ['track_id', 'VCL', 'VSL', 'VAP', 'LIN', 'WOB', 'ALH', 'BCF', 'track_length']
        st.dataframe(df_metrics[display_cols].round(2), use_container_width=True, height=300)
        
        # Distributions
        st.subheader("Metric Distributions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_metrics['VCL'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(df_metrics['VCL'].mean(), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('VCL (μm/s)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Curvilinear Velocity', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_metrics['VSL'], bins=15, edgecolor='black', alpha=0.7, color='coral')
            ax.axvline(df_metrics['VSL'].mean(), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('VSL (μm/s)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Straight-Line Velocity', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
        
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_metrics['LIN'], bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
            ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Progressive threshold')
            ax.set_xlabel('LIN', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Linearity', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
    
    with tab3:
        st.subheader("Physics-Based Trajectory Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # MSD analysis
            st.markdown("**Mean Squared Displacement (MSD)**")
            
            fig, ax = plt.subplots(figsize=(7, 6))
            
            for i, track in enumerate(tracks[:5]):
                traj_um = track.get_trajectory() * pixel_size
                lags, msd = compute_MSD(traj_um, max_lag=30)
                
                if len(lags) > 3:
                    time_lags = lags / 30
                    ax.plot(time_lags, msd, 'o-', alpha=0.6, label=f'Track {i+1}')
            
            # Fit for first track
            if len(tracks) > 0:
                traj_um = tracks[0].get_trajectory() * pixel_size
                lags, msd = compute_MSD(traj_um, max_lag=30)
                time_lags = lags / 30
                
                if len(lags) > 5:
                    fit_params = fit_MSD_diffusion(lags, msd, dt=1.0/30)
                    fit_msd = 4 * fit_params['D'] * time_lags**fit_params['alpha']
                    ax.plot(time_lags, fit_msd, 'r--', linewidth=2, 
                           label=f"Fit: α={fit_params['alpha']:.2f}")
            
            ax.set_xlabel('Time Lag (s)', fontsize=12)
            ax.set_ylabel('MSD (μm²)', fontsize=12)
            ax.set_title('MSD Analysis', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Diffusion parameters
            st.markdown("**Diffusion Parameters**")
            
            if len(tracks) > 0:
                traj_um = tracks[0].get_trajectory() * pixel_size
                lags, msd = compute_MSD(traj_um, max_lag=30)
                
                if len(lags) > 5:
                    fit_params = fit_MSD_diffusion(lags, msd, dt=1.0/30)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                    <h4>Sample Track Analysis</h4>
                    <p><b>Diffusion Coefficient (D):</b> {fit_params['D']:.2f} μm²/s</p>
                    <p><b>Anomalous Exponent (α):</b> {fit_params['alpha']:.2f}</p>
                    <p><b>Diffusion Regime:</b> {fit_params['regime']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if fit_params['alpha'] > 1.5:
                        st.success("✓ Ballistic motion detected - Active swimming!")
                    elif fit_params['alpha'] > 0.8:
                        st.info("Normal diffusive motion")
                    else:
                        st.warning("Subdiffusive motion")
    
    with tab4:
        st.subheader("Export Results")
        
        # CSV download
        csv = df_metrics.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=csv,
            file_name="sperm_metrics.csv",
            mime="text/csv"
        )
        
        # Summary text
        summary = f"""
        Sperm Quantification Analysis Summary
        =====================================
        
        Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Parameters:
        - Number of sperm: {n_sperm}
        - Duration: {duration}s
        - Simulation v0: {v0} μm/s
        
        Results:
        - Tracks detected: {len(tracks)}
        - Average VCL: {df_metrics['VCL'].mean():.2f} ± {df_metrics['VCL'].std():.2f} μm/s
        - Average VSL: {df_metrics['VSL'].mean():.2f} ± {df_metrics['VSL'].std():.2f} μm/s
        - Average LIN: {df_metrics['LIN'].mean():.2f} ± {df_metrics['LIN'].std():.2f}
        - Progressive motility: {(df_metrics['LIN'] > 0.5).sum()}/{len(tracks)} ({(df_metrics['LIN'] > 0.5).sum()/len(tracks)*100:.1f}%)
        
        Validation:
        - Expected VCL: {v0:.1f} μm/s
        - Measured VCL: {df_metrics['VCL'].mean():.1f} μm/s
        - Error: {abs(df_metrics['VCL'].mean() - v0)/v0*100:.1f}%
        """
        
        st.download_button(
            label="📄 Download Summary (TXT)",
            data=summary,
            file_name="analysis_summary.txt",
            mime="text/plain"
        )

else:
    # Welcome screen
    st.info("👈 Configure analysis parameters in the sidebar and click '🚀 Run Analysis' to start!")
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔬 Detection
        - Blob detection (DoG, LoG, DoH)
        - Adaptive thresholding
        - Size filtering
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Tracking
        - Kalman filtering
        - Hungarian algorithm
        - Gap handling
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Metrics
        - WHO-standardized (VCL, VSL, LIN, etc.)
        - Physics-based (MSD, diffusion)
        - Statistical analysis
        """)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This pipeline provides comprehensive analysis of sperm motility from videomicroscopy data.
    
    **Key Features:**
    - **Simulation**: Generate synthetic data using Active Brownian Particle physics
    - **Detection**: Multi-method blob detection optimized for sperm morphology
    - **Tracking**: Robust multi-object tracking with Kalman filtering
    - **Metrics**: WHO 2021 standardized motility parameters + physics analysis
    - **Visualization**: Publication-quality plots and interactive displays
    
    **Use Cases:**
    - Reproductive biology research
    - Microfluidic device validation
    - Sperm quality assessment
    - Biophysics studies
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sperm Quantification Pipeline v1.0 | Built with Streamlit</p>
    <p>For questions or issues, see README.md</p>
</div>
""", unsafe_allow_html=True)
