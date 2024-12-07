import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse



class MessagePassingLayer(MessagePassing):
	def __init__(self, latent_dim):
		super().__init__(aggr="add")
		self.edge_mlp = nn.Sequential(
			nn.Linear(latent_dim + 3, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.LayerNorm(latent_dim)
		)
		self.node_mlp = nn.Sequential(
			nn.Linear(latent_dim * 2, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.LayerNorm(latent_dim)
		)

	def forward(self, x, edge_index, edge_attr):
		# x: [num_particles, latent_dim]
		# edge_index: [2, num_edges]
		# edge_attr: [num_edges, 3]
		return self.propagate(edge_index, x=x, edge_attr=edge_attr)

	def message(self, x_j, edge_attr):
		# Concatenate node features (x_j) with edge features
		return self.edge_mlp(torch.cat([x_j, edge_attr], dim=-1))

	def update(self, aggr_out, x):
		# Concatenate aggregated messages with original node features
		return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


class OurModel(nn.Module):
	def __init__(self, in_dim, latent_dim, k, connectivity_radius):
		super().__init__()
		self.k = k
		self.connectivity_radius = connectivity_radius

		# Node encoder: MLP with two hidden layers and LayerNorm
		self.node_encoder = nn.Sequential(
			nn.Linear(in_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.LayerNorm(latent_dim)
		)

		# Message passing layers
		self.message_passing = nn.ModuleList([MessagePassingLayer(latent_dim) for _ in range(k)])

		# Node decoder: MLP without LayerNorm on output
		self.node_decoder = nn.Sequential(
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, 2)  # Output is a 2D vector for acceleration
		)

	def forward(self, x, positions):
		"""
		x: [batch_size, num_particles, in_dim] - Node features 
		positions: [batch_size, num_particles, 2] - Positions of particles
		"""
		batch_size = x.size(0)
		num_particles = x.size(1)

		# Encode node features (handle batch dimension)
		x = self.node_encoder(x.view(-1, x.size(-1)))  # [batch*num_particles, latent_dim]
		x = x.view(batch_size, num_particles, -1)

		# Compute pairwise distances for all batches at once
		batch_pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, num_particles, num_particles, 2]
		batch_x_dist = batch_pos_diff[..., 0]  # [batch, num_particles, num_particles]
		batch_y_dist = batch_pos_diff[..., 1]  # [batch, num_particles, num_particles]
		batch_euclidean = torch.sqrt(batch_x_dist**2 + batch_y_dist**2 + 1e-8)  # [batch, num_particles, num_particles]

		# Create adjacency matrices for all batches
		batch_adjacency = (batch_euclidean < self.connectivity_radius).float()  # [batch, num_particles, num_particles]

		# Process each batch item (still needed for sparse operations)
		accelerations = []
		for batch_idx in range(batch_size):
			# Convert to sparse format
			edge_index, _ = dense_to_sparse(batch_adjacency[batch_idx])

			# Create edge attributes efficiently
			edge_attr = torch.stack([
				batch_x_dist[batch_idx][edge_index[0], edge_index[1]],
				batch_y_dist[batch_idx][edge_index[0], edge_index[1]],
				batch_euclidean[batch_idx][edge_index[0], edge_index[1]]
			], dim=-1)

			# Perform message passing
			x_out = x[batch_idx]
			for layer in self.message_passing:
				x_out = layer(x_out, edge_index, edge_attr)

			# Decode to acceleration
			acc = self.node_decoder(x_out)  # [num_particles, 2]
			accelerations.append(acc)

		# Stack batch results
		return torch.stack(accelerations)