import h5py
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import shutil

def get_particle_counts(h5_file):
	"""Get number of particles for each simulation."""
	counts = {}
	for sim_idx in range(len(h5_file['positions'])):
		count = h5_file[f'positions/sim_{sim_idx}/position'][0].shape[0]
		counts[sim_idx] = count
	return counts

def create_split_file(args):
	"""Create a single split file (for parallel processing)."""
	input_path, output_path, indices, split_name = args
	
	with h5py.File(input_path, 'r') as src:
		with h5py.File(output_path / f"{split_name}.h5", 'w') as dst:
			# Copy attributes
			for key, value in src.attrs.items():
				dst.attrs[key] = value
			
			# Create groups
			positions_group = dst.create_group('positions')
			particle_types_group = dst.create_group('particle_types')
			
			# Copy selected simulations with new indices
			for new_idx, old_idx in enumerate(indices):
				# Copy positions
				src_pos = src[f'positions/sim_{old_idx}']
				positions_group.create_dataset(
					f'sim_{new_idx}/position',
					data=src_pos['position'][:],
					compression='gzip'
				)
				
				# Copy particle types
				src_types = src[f'particle_types/sim_{old_idx}']
				particle_types_group.create_dataset(
					f'sim_{new_idx}/particle_type',
					data=src_types['particle_type'][:],
					compression='gzip'
				)
	
	return split_name

def plot_particle_distributions(counts_by_split, output_path):
	"""Create histogram plot of particle counts for each split."""
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
	
	for ax, (split_name, counts) in zip([ax1, ax2, ax3], counts_by_split.items()):
		ax.hist(counts, bins=30, edgecolor='black')
		ax.set_title(f'{split_name.capitalize()} Split Particle Count Distribution')
		ax.set_xlabel('Number of Particles')
		ax.set_ylabel('Count')
		ax.set_xlim(0,2000)
		ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(output_path / 'particle_distributions.png', dpi=300, bbox_inches='tight')
	plt.close()

def create_splits(input_path: Path, output_path: Path, n_train: int, seed: int = None):
	"""Create train/valid/test splits based on particle counts."""
	if seed is not None:
		np.random.seed(seed)
		random.seed(seed)
	
	N_VALID = 30
	N_TEST = 30
	
	output_path.mkdir(parents=True, exist_ok=True)
	
	# Copy metadata
	if (input_path.parent / 'metadata.json').exists():
		shutil.copy2(input_path.parent / 'metadata.json', output_path / 'metadata.json')
	
	print("Analyzing particle counts...")
	with h5py.File(input_path, 'r') as src:
		counts = get_particle_counts(src)
		
		# Split indices based on particle counts
		small_indices = [idx for idx, count in counts.items() if count < 1000]
		medium_indices = [idx for idx, count in counts.items() if 1000 <= count < 1500]
		large_indices = [idx for idx, count in counts.items() if count >= 1500]
		
		print(f"\nFound simulations by particle count:")
		print(f"<1000 particles: {len(small_indices)}")
		print(f"1000-1500 particles: {len(medium_indices)}")
		print(f">1500 particles: {len(large_indices)}")
		
		# Verify we have enough simulations
		if len(small_indices) < n_train:
			raise ValueError(f"Not enough simulations with <1000 particles. Requested {n_train}, found {len(small_indices)}")
		if len(medium_indices) < N_VALID:
			raise ValueError(f"Not enough simulations with 1000-1500 particles. Need {N_VALID}, found {len(medium_indices)}")
		if len(large_indices) < N_TEST:
			raise ValueError(f"Not enough simulations with >1500 particles. Need {N_TEST}, found {len(large_indices)}")
		
		# Sample indices for each split
		train_indices = np.random.choice(small_indices, size=n_train, replace=False)
		valid_indices = np.random.choice(medium_indices, size=N_VALID, replace=False)
		test_indices = np.random.choice(large_indices, size=N_TEST, replace=False)
		
		# Prepare arguments for parallel processing
		split_args = [
			(input_path, output_path, train_indices, 'train'),
			(input_path, output_path, valid_indices, 'valid'),
			(input_path, output_path, test_indices, 'test')
		]
		
		# Create splits in parallel
		print("\nCreating splits in parallel...")
		with ProcessPoolExecutor(max_workers=3) as executor:
			results = list(executor.map(create_split_file, split_args))
		
		print("\nAll splits created successfully!")
		
		# Create visualization
		counts_by_split = {
			'train': [counts[idx] for idx in train_indices],
			'valid': [counts[idx] for idx in valid_indices],
			'test': [counts[idx] for idx in test_indices]
		}
		plot_particle_distributions(counts_by_split, output_path)
		print(f"\nParticle distribution plots saved to {output_path}/particle_distributions.png")

def main():
	parser = argparse.ArgumentParser(description='Create train/valid/test splits based on particle counts')
	parser.add_argument('input_file', type=Path, help='Input train.h5 file')
	parser.add_argument('output_dir', type=Path, help='Output directory for new splits')
	parser.add_argument('n_train', type=int, help='Number of training simulations to sample')
	parser.add_argument('--seed', type=int, help='Random seed (optional)', default=None)
	
	args = parser.parse_args()
	
	if args.seed is None:
		args.seed = random.randint(0, 2**32 - 1)
		print(f"Using random seed: {args.seed}")
	
	create_splits(args.input_file, args.output_dir, args.n_train, args.seed)

if __name__ == '__main__':
	main()