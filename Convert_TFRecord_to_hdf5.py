import tensorflow as tf
import h5py
import numpy as np
import json
import os
from typing import Dict, Any, Tuple
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import psutil

def read_metadata(data_path: str) -> Dict[str, Any]:
    """Read metadata.json file from the dataset directory."""
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def convert_to_tensor(x, encoded_dtype):
    """Convert multiple byte strings to numpy array."""
    out = []
    for el in x:
        out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    return np.concatenate(out) if len(out) > 1 else out[0]

def parse_serialized_simulation_example(example_proto: bytes, metadata: Dict) -> Dict[str, np.ndarray]:
    """Parse a single serialized example using the format from the paper."""
    sequence_features = {
        'position': tf.io.VarLenFeature(tf.string),
    }
    
    context_features = {
        'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'particle_type': tf.io.VarLenFeature(tf.string)
    }
    
    # Parse both context and sequence features
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )
    
    # Convert positions - handle multiple tensors
    position_bytes = parsed_features['position'].values
    positions = convert_to_tensor(position_bytes, np.float32)
    sequence_length = metadata['sequence_length'] + 1
    
    # Count number of particles based on positions size
    n_particles = len(positions) // (sequence_length * metadata['dim'])
    
    # Reshape positions to [sequence_length, n_particles, dim]
    positions = positions.reshape(sequence_length, n_particles, metadata['dim'])
    
    # Convert particle types
    particle_bytes = context['particle_type'].values
    particle_types = convert_to_tensor(particle_bytes, np.int64)
    
    return {
        'position': positions,
        'particle_type': particle_types
    }

def process_record(args: Tuple) -> Tuple[int, Dict[str, np.ndarray]]:
    """Process a single record for parallel execution."""
    i, record_bytes, metadata = args
    data = parse_serialized_simulation_example(record_bytes, metadata)
    return i, data

def convert_tfrecord_to_hdf5(input_path: str, output_path: str, n_workers: int) -> None:
    """Convert all TFRecord files in a directory to HDF5 format using parallel processing."""
    os.makedirs(output_path, exist_ok=True)
    metadata = read_metadata(input_path)
    
    # Copy metadata
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    for split in ['train', 'valid', 'test']:
        tfrecord_path = os.path.join(input_path, f'{split}.tfrecord')
        if not os.path.exists(tfrecord_path):
            continue
        
        h5_path = os.path.join(output_path, f'{split}.h5')
        print(f"\nConverting {split} split...")
        
        # Read all records into memory
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        records = [(i, record.numpy(), metadata) for i, record in enumerate(dataset)]
        n_records = len(records)
        
        # Process records in parallel
        with Pool(n_workers) as pool:
            with h5py.File(h5_path, 'w') as h5f:
                # Store metadata attributes
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple)):
                        value = np.array(value)
                    h5f.attrs[key] = value
                
                # Create groups
                position_group = h5f.create_group('positions')
                particle_type_group = h5f.create_group('particle_types')
                
                # Process records with progress bar
                for i, data in tqdm(pool.imap(process_record, records), total=n_records, desc=f"Processing {split}"):
                    position_group.create_dataset(
                        f'sim_{i}/position', 
                        data=data['position'],
                        compression='gzip'
                    )
                    particle_type_group.create_dataset(
                        f'sim_{i}/particle_type', 
                        data=data['particle_type'],
                        compression='gzip'
                    )

def validate_conversion(h5_path: str) -> None:
    """Validate the conversion results."""
    print(f"\nValidating {os.path.basename(h5_path)}:")
    with h5py.File(h5_path, 'r') as f:
        sim_key = 'sim_0'
        positions = f[f'positions/{sim_key}/position'][:]
        types = f[f'particle_types/{sim_key}/particle_type'][:]
        
        print(f"First simulation:")
        print(f"Position shape: {positions.shape}")
        print(f"Position range: [{positions.min():.6f}, {positions.max():.6f}]")
        print(f"Particle types shape: {types.shape}")
        print(f"Unique particle types: {np.unique(types)}")

def main():
    parser = argparse.ArgumentParser(description='Convert TFRecord particle simulation data to HDF5 format')
    parser.add_argument('input_dir', help='Input directory containing TFRecord files and metadata.json')
    parser.add_argument('output_dir', help='Output directory for HDF5 files')
    parser.add_argument('--workers', type=int, default=max(1, psutil.cpu_count() - 3),
                      help='Number of worker processes (default: number of CPUs - 3)')
    args = parser.parse_args()
    
    print(f"Using {args.workers} worker processes")
    convert_tfrecord_to_hdf5(args.input_dir, args.output_dir, args.workers)
    
    # Validate each split
    for split in ['train', 'valid', 'test']:
        h5_path = os.path.join(args.output_dir, f'{split}.h5')
        if os.path.exists(h5_path):
            validate_conversion(h5_path)

if __name__ == '__main__':
    main()