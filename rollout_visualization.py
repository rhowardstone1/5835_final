import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import h5py
from pathlib import Path
from typing import Optional, Dict, Union


def save_rollout(
    save_path: Path,
    positions: np.ndarray,
    accelerations: np.ndarray,
    particle_types: np.ndarray,
    metadata: Optional[Dict] = None
):
    """Save rollout data to HDF5."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('positions', data=positions, compression='gzip')
        f.create_dataset('accelerations', data=accelerations, compression='gzip')
        f.create_dataset('particle_types', data=particle_types, compression='gzip')
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

def load_rollout(load_path: Path) -> Dict[str, Union[np.ndarray, Dict]]:
    """Load rollout data from HDF5."""
    with h5py.File(load_path, 'r') as f:
        data = {
            'positions': f['positions'][:],
            'accelerations': f['accelerations'][:],
            'particle_types': f['particle_types'][:],
            'metadata': dict(f.attrs)
        }
    return data

def calculate_kinetic_energy(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Calculate kinetic energy (v^2) for each timestep.
    Args:
        positions: [timesteps, n_particles, dims]
        dt: time step size
    Returns:
        [timesteps] array of total kinetic energy
    """
    velocities = (positions[1:] - positions[:-1]) / dt
    kinetic = np.sum(velocities**2, axis=(1,2))
    return kinetic

def create_animation_comparison(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
    particle_types: np.ndarray,
    dt: float,
    save_path: Optional[Path] = None,
    bounds: Optional[np.ndarray] = None,
    fps: int = 30,
    title: str = "Particle Simulation Comparison"
):
    """Create animation comparing prediction and ground truth with kinetic energy plots."""
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # Particle position plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Kinetic energy plots
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Fixed bounds for particle position plots
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.set_title("Prediction")
    ax2.set_title("Ground Truth")
    
    # Calculate kinetic energies
    pred_ke = calculate_kinetic_energy(pred_positions, dt)
    true_ke = calculate_kinetic_energy(true_positions, dt)
    
    # Setup kinetic energy plots
    time_points = np.arange(len(pred_ke)) * dt
    
    # Initialize lines for kinetic energy
    line1, = ax3.plot([], [], 'b-', label='Predicted KE')
    line2, = ax4.plot([], [], 'b-', label='Ground Truth KE')
    
    # Set up KE plot formatting
    for ax, label in [(ax3, 'Predicted'), (ax4, 'Ground Truth')]:
        ax.set_xlabel('Time')
        ax.set_ylabel('Kinetic Energy (v²)')
        ax.set_title(f'{label} Kinetic Energy')
        ax.grid(True)
        ax.set_xlim(0, time_points[-1])
    
    # Set different y-limits for each KE plot
    ax3.set_ylim(0, pred_ke.max() * 1.1)
    ax4.set_ylim(0, true_ke.max() * 1.1)
    
    # Create scatter plots for particles
    scatter1 = ax1.scatter(pred_positions[0, :, 0], pred_positions[0, :, 1], 
                          c=particle_types, cmap='tab10')
    scatter2 = ax2.scatter(true_positions[0, :, 0], true_positions[0, :, 1], 
                          c=particle_types, cmap='tab10')
    
    fig.suptitle(title, fontsize=16)
    
    def update(frame):
        # Update particle positions
        scatter1.set_offsets(pred_positions[frame, :, :2])
        scatter2.set_offsets(true_positions[frame, :, :2])
        
        # Update kinetic energy plots
        line1.set_data(time_points[:frame], pred_ke[:frame])
        line2.set_data(time_points[:frame], true_ke[:frame])
        
        if frame > 0:
            ax3.set_ylim(0, pred_ke[:frame].max() * 1.1)
            ax4.set_ylim(0, true_ke[:frame].max() * 1.1)
        
        fig.suptitle(f"{title} - Frame {frame} (t={frame*dt:.3f}s)", fontsize=16)
        return scatter1, scatter2, line1, line2
    
    anim = FuncAnimation(
        fig, update, frames=len(pred_positions),
        interval=1000/fps, blit=True
    )
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'))
        anim.save(save_path, writer=writer)
        plt.close()
    else:
        plt.show()

def plot_comparison_states(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
    particle_types: np.ndarray,
    save_path: Optional[Path] = None,
    bounds: Optional[np.ndarray] = None,
    timesteps: list = [-1],  # Can specify multiple timesteps to plot
    title: str = "State Comparison"
):
    """Create static comparison plots at specified timesteps."""
    n_timesteps = len(timesteps)
    fig, axes = plt.subplots(n_timesteps, 2, figsize=(20, 10*n_timesteps))
    if n_timesteps == 1:
        axes = axes.reshape(1, -1)
    
    # Fixed bounds for all particle plots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
    
    for i, t in enumerate(timesteps):
        axes[i, 0].scatter(pred_positions[t, :, 0], pred_positions[t, :, 1], 
                          c=particle_types, cmap='tab10')
        axes[i, 1].scatter(true_positions[t, :, 0], true_positions[t, :, 1], 
                          c=particle_types, cmap='tab10')
        
        axes[i, 0].set_title(f"Prediction (t={t})")
        axes[i, 1].set_title(f"Ground Truth (t={t})")
    
    plt.suptitle(title, fontsize=16)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()