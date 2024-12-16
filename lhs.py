import numpy as np
from pyDOE2 import lhs

def latin_hypercube_sampling(param_ranges, n_samples):
    """
    Perform Latin Hypercube Sampling (LHS) for the given parameter ranges.

    Parameters:
        param_ranges (list of tuples): List of parameter ranges as (min, max) tuples.
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: An array of shape (n_samples, len(param_ranges)) with sampled points.
    """
    # Number of parameters (dimensions)
    n_params = len(param_ranges)

    # Generate LHS samples in the unit hypercube
    lhs_samples = lhs(n_params, samples=n_samples)

    # Scale samples to the provided parameter ranges
    scaled_samples = np.zeros_like(lhs_samples)
    for i, (low, high) in enumerate(param_ranges):
        scaled_samples[:, i] = low + (high - low) * lhs_samples[:, i]

    return scaled_samples

# Example usage
if __name__ == "__main__":
    # Define hyper-parameter ranges as (min, max) tuples
    # param_ranges = [(3, 7), (3, 12)]  # For example: parameter 1 in [0, 1], parameter 2 in [10, 100]

    # # Number of samples to generate
    # n_samples = 8

    # # Perform LHS
    # samples = latin_hypercube_sampling(param_ranges, n_samples)

    # # Print the generated samples
    # print("Generated Samples (C, MPS):")
    # print([ [round(a), round(b)] for a,b in samples])

    samples = [[4, 4], [5, 3], [3, 8], [6, 6], [6, 10], [4, 11], [5, 10], [7, 7]]
    samples = sorted(samples, key=lambda x: -x[1])
    print(samples)

    x=[[],[],[],[]]
    i = 0
    for sample in samples:
        x[i].append(sample)
        i = (i+1) % 4

    print(x)