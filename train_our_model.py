import torch, os, config
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import ParticleDataset, collate_fn  # Assuming this is defined in data.py --woooo!
#from data import PrecomputedParticleDataset, collate_fn
import our_model as om
from numpy import inf
from tqdm import tqdm


##comment out for real dataset:

#num_particles = 500
#
#from torch.utils.data import Dataset
#
#class FakeDataset(Dataset):
#	def __init__(self):
#		super().__init__()
#		
#	def __len__(self):
#		return 100 * (config.SEQUENCE_LENGTH - config.INPUT_SEQUENCE_LENGTH)
#
#	def __getitem__(self, idx):
#		x = torch.rand((num_particles, feature_dim))
#		p = torch.rand((num_particles, 2))
#		y = torch.rand((num_particles, 2))
#
#		return p, x, y
	




feature_dim = (config.DIM +				 # current position (x,y)
			  (config.INPUT_SEQUENCE_LENGTH-1)*config.DIM +  # velocity history
					  1 +						   # particle type
					  2*config.DIM)				 # distances to walls

device = torch.device(config.DEVICE)

model = om.OurModel(
	in_dim=feature_dim, 
	latent_dim=config.LATENT_DIM, 
	k=config.NUM_MP_STEPS, 
	connectivity_radius=config.CONNECTIVITY_RADIUS
).to(device)


#train_dataset = FakeDataset()
train_dataset = ParticleDataset()
#train_dataset = PrecomputedParticleDataset()
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)

print(f'\nYou may expect: {len(train_loader)} iterations per epoch:\n')

best_loss = inf

os.makedirs(config.OUT_DIR, exist_ok = True)




# Calculate gamma for exponential decay from 1e-4 to 1e-6 over 20M steps
steps_per_epoch = len(train_loader)
total_steps = config.MAX_EPOCHS * steps_per_epoch
gamma = (1e-6 / 1e-4) ** (1 / total_steps)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
criterion = nn.MSELoss()

for epoch in range(config.MAX_EPOCHS):
	model.train()  # Set the model to training mode
	total_loss = 0.0

	for batch_idx, (positions, node_features, target_accelerations, padding_masks) in tqdm(enumerate(train_loader)):
		positions = positions.to(device)
		node_features = node_features.to(device)
		target_accelerations = target_accelerations.to(device)
		padding_masks = padding_masks.to(device)

		optimizer.zero_grad()
		predicted_accelerations = model(node_features, positions)

		# Apply mask to ignore padded particles in loss
		loss = criterion(
			predicted_accelerations[padding_masks], 
			target_accelerations[padding_masks]
		)

		loss.backward()
		optimizer.step()
		scheduler.step()
		total_loss += loss.item()
	
	avg_loss = total_loss / len(train_loader)
	
	if avg_loss < best_loss:
		best_loss = avg_loss
		torch.save({
			'epoch': epoch + 1,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
			'best_loss': best_loss,
		}, config.OUT_DIR / f"best_model.pth")
	
	print(f"\nEpoch [{epoch+1}/{config.MAX_EPOCHS}], Loss: {avg_loss:.4f}")

	if (epoch + 1) % 1 == 0 or epoch + 1 == config.MAX_EPOCHS:
		torch.save({
			'epoch': epoch + 1,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
			'best_loss': best_loss,
		}, config.OUT_DIR / f"model_epoch_{epoch+1}.pth")

print("Training complete.")

