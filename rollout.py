import torch
import numpy as np
import config
from pathlib import Path
from typing import Optional, Tuple


class RolloutSimulator:
	def __init__(self, model, device=None):
		"""
		Initialize rollout simulator with trained model.
		Args:
			model: Trained model instance
			device: torch device (defaults to config.DEVICE)
		"""
		self.model = model
		self.device = device or torch.device(config.DEVICE)
		self.model.to(self.device)
		self.model.eval()

	def _compute_velocities(self, positions_window: np.ndarray) -> np.ndarray:
		"""
		Compute and normalize velocities from position sequence.
		Args:
			positions_window: Array of shape [sequence_length, n_particles, dims]
		Returns:
			Normalized velocities of shape [n_particles, sequence_length-1, dims]
		"""
		velocities = []
		for i in range(len(positions_window)-1):
			velocity = (positions_window[i+1] - positions_window[i]) / config.DT
			velocities.append(velocity)
		
		velocities = np.stack(velocities, axis=1)  # [n_particles, sequence_length-1, dims]
		
		# Normalize using stored statistics
		velocities = (velocities - np.array(config.VELOCITY_MEAN)) / np.array(config.VELOCITY_STD)
		return velocities

	def _compute_wall_distances(self, positions: np.ndarray) -> np.ndarray:
		"""
		Compute distances to walls, clipped by connectivity radius.
		Args:
			positions: Array of shape [n_particles, dims]
		Returns:
			Wall distances of shape [n_particles, 2*dims]
		"""
		bounds = np.array(config.BOUNDS)
		lower_dists = positions - bounds[:, 0]
		upper_dists = bounds[:, 1] - positions
		return np.clip(
			np.column_stack([lower_dists, upper_dists]),
			0,
			config.CONNECTIVITY_RADIUS
		)

	def _construct_node_features(
		self,
		current_pos: np.ndarray,
		velocities: np.ndarray,
		particle_types: np.ndarray
	) -> np.ndarray:
		"""
		Construct node features exactly as in training.
		"""
		n_particles = len(particle_types)
		feature_dim = (config.DIM +				  # current position
					  (config.INPUT_SEQUENCE_LENGTH-1)*config.DIM +  # velocity history
					  1 +							# particle type
					  2*config.DIM)				  # wall distances
		
		node_features = np.zeros((n_particles, feature_dim))
		curr_idx = 0
		
		# Current position
		node_features[:, curr_idx:curr_idx+config.DIM] = current_pos
		curr_idx += config.DIM
		
		# Velocity history (reshaped to flatten sequence)
		velocities = velocities.reshape(velocities.shape[0], -1)
		node_features[:, curr_idx:curr_idx+(config.INPUT_SEQUENCE_LENGTH-1)*config.DIM] = velocities
		curr_idx += (config.INPUT_SEQUENCE_LENGTH-1)*config.DIM
		
		# Particle type
		node_features[:, curr_idx] = particle_types
		curr_idx += 1
		
		# Wall distances
		node_features[:, curr_idx:] = self._compute_wall_distances(current_pos)
		
		return node_features

	def step(
		self,
		positions_window: np.ndarray,
		particle_types: np.ndarray
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Perform single prediction step with corrected dynamics.
		Args:
			positions_window: Array of shape [input_sequence_length, n_particles, dims]
			particle_types: Array of shape [n_particles]
		Returns:
			Tuple of (next_position, acceleration)
		"""
		with torch.no_grad():
			# Compute velocities and features
			velocities = self._compute_velocities(positions_window)
			current_pos = positions_window[-2]  # Second to last position
			node_features = self._construct_node_features(current_pos, velocities, particle_types)

			# Convert to tensors
			current_pos = torch.FloatTensor(current_pos).to(self.device)
			node_features = torch.FloatTensor(node_features).to(self.device)

			# Get model prediction (add batch dimension)
			pred_acceleration = self.model(
				node_features.unsqueeze(0),
				current_pos.unsqueeze(0)
			)[0]  # Remove batch dimension

			# Convert to numpy and denormalize
			acceleration = pred_acceleration.cpu().numpy()
			acceleration = acceleration * np.array(config.ACCELERATION_STD) + np.array(config.ACCELERATION_MEAN)

			# Semi-implicit Euler integration with boundary conditions
			current_velocity = (positions_window[-1] - positions_window[-2]) / config.DT
			next_velocity = current_velocity + acceleration * config.DT
			next_position = positions_window[-1] + next_velocity * config.DT

			# Apply boundary conditions
			#bounds = np.array(config.BOUNDS)
			#for dim in range(config.DIM):
			#	# Check lower bound
			#	below_bound = next_position[:, dim] < bounds[dim, 0]
			#	next_position[below_bound, dim] = bounds[dim, 0]
			#	next_velocity[below_bound, dim] = 0  # Stop velocity at boundary
			#
			#	# Check upper bound
			#	above_bound = next_position[:, dim] > bounds[dim, 1]
			#	next_position[above_bound, dim] = bounds[dim, 1]
			#	next_velocity[above_bound, dim] = 0  # Stop velocity at boundary

			return next_position, acceleration
		
	def rollout(
		self, 
		initial_positions: np.ndarray,
		particle_types: np.ndarray,
		n_steps: int
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Perform rollout starting from initial conditions.
		Args:
			initial_positions: Array of shape [input_sequence_length, n_particles, dims]
			particle_types: Array of shape [n_particles]
			n_steps: Number of steps to roll out
		Returns:
			Tuple of (positions, accelerations) containing full trajectory
		"""
		if len(initial_positions) != config.INPUT_SEQUENCE_LENGTH:
			raise ValueError(
				f"Expected {config.INPUT_SEQUENCE_LENGTH} initial positions, "
				f"got {len(initial_positions)}"
			)
		
		# Initialize trajectory storage
		n_particles = len(particle_types)
		positions = np.zeros((n_steps + config.INPUT_SEQUENCE_LENGTH, n_particles, config.DIM))
		accelerations = np.zeros((n_steps, n_particles, config.DIM))
		
		# Set initial positions
		positions[:config.INPUT_SEQUENCE_LENGTH] = initial_positions
		
		# Perform rollout
		for step in range(n_steps):
			window_start = step
			window_end = step + config.INPUT_SEQUENCE_LENGTH
			position_window = positions[window_start:window_end]
			
			next_position, acceleration = self.step(position_window, particle_types)
			
			positions[window_end] = next_position
			accelerations[step] = acceleration
			
		return positions, accelerations