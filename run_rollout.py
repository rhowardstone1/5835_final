import torch
import config
from pathlib import Path
import h5py
import numpy as np
from rollout import RolloutSimulator
from rollout_visualization import (
    save_rollout, 
    create_animation_comparison, 
    plot_comparison_states
)
import our_model as om

def load_simulation_data(data_path: Path, sim_idx: int):
    """Load full simulation data including ground truth."""
    with h5py.File(data_path, 'r') as f:
        positions = f[f'positions/sim_{sim_idx}/position'][:]
        particle_types = f[f'particle_types/sim_{sim_idx}/particle_type'][:]
    return positions, particle_types

def calculate_mse(pred_positions: np.ndarray, true_positions: np.ndarray):
    """Calculate MSE between predicted and true positions."""
    return np.mean((pred_positions - true_positions) ** 2)

def main():
    # Create model instance
    feature_dim = (config.DIM +                 # current position (x,y)
                  (config.INPUT_SEQUENCE_LENGTH-1)*config.DIM +  # velocity history
                  1 +                           # particle type
                  2*config.DIM)                 # distances to walls
    
    model = om.OurModel(
        in_dim=feature_dim, 
        latent_dim=config.LATENT_DIM, 
        k=config.NUM_MP_STEPS, 
        connectivity_radius=config.CONNECTIVITY_RADIUS
    )
    
    # Load trained model
    checkpoint_path = config.OUT_DIR / "best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
    
    # Create simulator
    simulator = RolloutSimulator(model)
    
    # Load full simulation data for comparison
    data_path = config.DATA_DIR / "valid.h5"
    test_sim_idx = 0  # Can modify to test different simulations
    true_positions, particle_types = load_simulation_data(data_path, test_sim_idx)
    
    # Get initial conditions
    initial_positions = true_positions[:config.INPUT_SEQUENCE_LENGTH]
    
    # Perform rollout for same number of steps as ground truth
    n_steps = len(true_positions) - config.INPUT_SEQUENCE_LENGTH
    print(f"Starting rollout for {n_steps} steps...")
    
    pred_positions, accelerations = simulator.rollout(
        initial_positions, 
        particle_types,
        n_steps
    )
    
    # Calculate MSE
    mse = calculate_mse(pred_positions[config.INPUT_SEQUENCE_LENGTH:], 
                       true_positions[config.INPUT_SEQUENCE_LENGTH:])
    print(f"Overall MSE: {mse:.6f}")
    
    # Create output directory
    rollout_dir = config.OUT_DIR / "rollouts"
    rollout_dir.mkdir(exist_ok=True)
    
    # Save rollout data
    save_path = rollout_dir / "valid_rollout.h5"
    print(f"Saving rollout data to {save_path}")
    save_rollout(
        save_path,
        pred_positions,
        accelerations,
        particle_types,
        metadata={
            'n_steps': n_steps,
            'mse': float(mse),
            'test_sim_idx': test_sim_idx
        }
    )
    
    # Create side-by-side animation
    print("Creating comparison animation...")
    create_animation_comparison(
        pred_positions[config.INPUT_SEQUENCE_LENGTH:],
        true_positions[config.INPUT_SEQUENCE_LENGTH:],
        particle_types,
		config.DT,
        save_path=rollout_dir / "comparison_rollout.mp4",
        bounds=config.BOUNDS,
        title=f"Test Simulation {test_sim_idx} (MSE: {mse:.6f})"
    )
    
    # Plot comparison at multiple timesteps
    print("Saving comparison plots...")
    timesteps = [0, n_steps//4, n_steps//2, -1]  # Plot at 0%, 25%, 50%, and 100%
    plot_comparison_states(
        pred_positions[config.INPUT_SEQUENCE_LENGTH:],
        true_positions[config.INPUT_SEQUENCE_LENGTH:],
        particle_types,
        save_path=rollout_dir / "comparison_states.png",
        bounds=config.BOUNDS,
        timesteps=timesteps,
        title=f"Test Simulation {test_sim_idx} States (MSE: {mse:.6f})"
    )
    
    print("Done!")

if __name__ == "__main__":
    main()