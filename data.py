import h5py, torch, config
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

def collate_fn(batch):
	"""
	Pads batch elements to max particle count in batch.
	Args:
		batch: List of (positions, node_features, accelerations) tuples
	Returns:
		Batched and padded tensors
	"""
	max_particles = max(pos.shape[0] for pos, _, _ in batch)
	
	padded_positions = []
	padded_features = [] 
	padded_accels = []
	padding_masks = []
	
	for positions, node_features, accelerations in batch:
		n_particles = positions.shape[0]
		
		if n_particles < max_particles:
			# Create padding mask (1 for real particles, 0 for padding)
			mask = torch.ones(max_particles, dtype=torch.bool)
			mask[n_particles:] = 0
			
			# Pad each tensor
			positions = torch.nn.functional.pad(positions, (0, 0, 0, max_particles - n_particles))
			node_features = torch.nn.functional.pad(node_features, (0, 0, 0, max_particles - n_particles))
			accelerations = torch.nn.functional.pad(accelerations, (0, 0, 0, max_particles - n_particles))
		else:
			mask = torch.ones(max_particles, dtype=torch.bool)
			
		padded_positions.append(positions)
		padded_features.append(node_features)
		padded_accels.append(accelerations)
		padding_masks.append(mask)
			
	return (
		torch.stack(padded_positions),
		torch.stack(padded_features),
		torch.stack(padded_accels),
		torch.stack(padding_masks)
	)


def generate_simulation_noise(args):
	"""Generate noise for a single simulation."""
	sim_idx, positions, noise_std = args
	sequence_length = len(positions)
	noise_shape = (sequence_length, positions.shape[1], positions.shape[2])  # time, particles, dims
	noise = np.random.randn(*noise_shape) * noise_std
	return sim_idx, np.cumsum(noise, axis=0)

class ParticleDataset:
	def __init__(self, split='train', n_workers=None):
		super().__init__()
		self.split = split
		self.h5_path = config.DATA_DIR / f"{split}.h5"
		self.h5_file = h5py.File(self.h5_path, 'r')
		self.num_sims = len(self.h5_file['positions'])
		
		if not n_workers:
			n_workers = max(1, cpu_count()-3)  # Leave 3 CPUs un-used
		
		# Generate noise for entire simulation (upfront)
		if split == 'train':  # Only add noise during training
			
			print(f"Generating noise for training dataset using {n_workers} workers..")
			# Load all positions at once
			args = []
			for sim_idx in range(self.num_sims):
				positions = self.h5_file[f'positions/sim_{sim_idx}/position'][:]
				args.append((sim_idx, positions, config.NOISE_STD))
			
			# Generate noise in parallel
			self.sim_noises = {}
			with Pool(n_workers) as pool:
				for sim_idx, noise in tqdm(
					pool.imap(generate_simulation_noise, args),
					total=len(args),
					desc="Generating noise"
				):
					self.sim_noises[sim_idx] = noise
	
	def __len__(self):
		return self.num_sims * (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
	
	def __getitem__(self, idx):
		sim_idx = idx // (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
		step_idx = idx % (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
		
		positions = self.h5_file[f'positions/sim_{sim_idx}/position'][
			step_idx:step_idx + config.INPUT_SEQUENCE_LENGTH + 1
		]
		
		if self.split == 'train':
			noise_slice = self.sim_noises[sim_idx][
				step_idx:step_idx + config.INPUT_SEQUENCE_LENGTH + 1
			]
			positions = positions + noise_slice
		
		particle_types = self.h5_file[f'particle_types/sim_{sim_idx}/particle_type'][:]
		
		# Calculate velocities from position differences
		velocities = []
		for i in range(config.INPUT_SEQUENCE_LENGTH - 1):
			velocity = (positions[i+1] - positions[i]) / config.DT
			velocities.append(velocity)
		
		# Stack velocities into array [n_particles, input_length-1, dims]
		velocities = np.stack(velocities, axis=1)
		
		# Normalize velocities
		velocities = (velocities - np.array(config.VELOCITY_MEAN)) / np.array(config.VELOCITY_STD)
		
		# Calculate target acceleration from last 3 positions
		vel_next = (positions[-1] - positions[-2]) / config.DT
		vel_curr = (positions[-2] - positions[-3]) / config.DT
		acceleration = (vel_next - vel_curr) / config.DT
		
		# Normalize acceleration
		acceleration = (acceleration - np.array(config.ACCELERATION_MEAN)) / np.array(config.ACCELERATION_STD)
		
		# Calculate wall distances efficiently
		bounds = np.array(config.BOUNDS)
		current_pos = positions[-2]
		lower_dists = current_pos - bounds[:, 0]
		upper_dists = bounds[:, 1] - current_pos
		dist_to_walls = np.clip(
			np.column_stack([lower_dists, upper_dists]),
			0,
			config.CONNECTIVITY_RADIUS
		)
		
		# Create node features
		n_particles = len(particle_types)
		feature_dim = (config.DIM +					 # current position (x,y)
					  (config.INPUT_SEQUENCE_LENGTH-1)*config.DIM +  # velocity history
					  1 +							   # particle type
					  2*config.DIM)					 # distances to walls
		
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
		node_features[:, curr_idx:] = dist_to_walls
		
		return torch.FloatTensor(current_pos), torch.FloatTensor(node_features), torch.FloatTensor(acceleration)
	
	def __del__(self):
		self.h5_file.close()

		
		
		
		
		
		
		
#Everything from here down could probably be deleted :/

class PrecomputedParticleDataset:
	def __init__(self, split='train'):
		self.split = split
		self.h5_path = config.DATA_DIR / f"{split}_precomputed.h5"
		self.h5_file = h5py.File(self.h5_path, 'r')
		self.num_sims = len(self.h5_file['positions'])
	
	def __len__(self):
		return self.num_sims * (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
	
	def __getitem__(self, idx):
		sim_idx = idx // (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
		step_idx = idx % (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
		
		current_pos = self.h5_file[f'positions/sim_{sim_idx}/position'][step_idx + config.INPUT_SEQUENCE_LENGTH - 1]
		node_features = self.h5_file[f'node_features/sim_{sim_idx}'][step_idx]
		acceleration = self.h5_file[f'accelerations/sim_{sim_idx}'][step_idx]
		
		return torch.FloatTensor(current_pos), torch.FloatTensor(node_features), torch.FloatTensor(acceleration)
	
	def __del__(self):
		self.h5_file.close()

def precompute_datasets(n_workers=None):
	"""Precompute features for all splits."""
	for split in ['valid', 'test', 'train']:
		input_path = config.DATA_DIR / f"{split}.h5"
		if not input_path.exists():
			continue
		
		output_path = config.DATA_DIR / f"{split}_precomputed.h5"
		if output_path.exists():
			print(f"\nPrecomputed {split} file already exists at {output_path}. Skipping...")
		else:
			print(f"\nPrecomputing {split} split...")

			dataset = ParticleDataset(split)
			print("Created ParticleDataset")

			with h5py.File(output_path, 'w') as out_file:
				# Copy original data
				with h5py.File(input_path, 'r') as in_file:
					for key in ['positions', 'particle_types']:
						in_file.copy(key, out_file)
				print("Copied original data")

				# Create feature groups
				features = out_file.create_group('node_features')
				accelerations = out_file.create_group('accelerations')
				print("Created feature groups")

				for sim_idx in tqdm(range(dataset.num_sims)):
					sim_features = []
					sim_accels = []

					steps = config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH
					for step in range(steps):
						idx = sim_idx * steps + step
						_, node_feats, accel = dataset[idx]
						sim_features.append(node_feats.numpy())
						sim_accels.append(accel.numpy())

					features.create_dataset(
						f'sim_{sim_idx}',
						data=np.stack(sim_features),
						compression='gzip'
					)
					accelerations.create_dataset(
						f'sim_{sim_idx}',
						data=np.stack(sim_accels),
						compression='gzip'
					)

			# Validate conversion
			if split != "train": # Skip validation for train due to noise
				validate_conversion(split, output_path)

def validate_conversion(split: str, output_path: Path):
	"""Compare original and precomputed datasets."""
	orig_dataset = ParticleDataset(split)
	precomp_dataset = PrecomputedParticleDataset(split)
	
	# Check random samples
	for _ in range(5):
		idx = np.random.randint(len(orig_dataset))
		orig_out = orig_dataset[idx]
		precomp_out = precomp_dataset[idx]
		
		for o, p in zip(orig_out, precomp_out):
			assert torch.allclose(o, p, rtol=1e-5), f"Mismatch at index {idx}"
	
	print(f"Validated {split} conversion successfully")

if __name__ == "__main__":
	precompute_datasets()