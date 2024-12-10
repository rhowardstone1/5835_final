import torch, os, config, argparse
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import ParticleDataset, collate_fn
import our_model as om
from numpy import inf
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
args = parser.parse_args()

feature_dim = (config.DIM +                 # current position (x,y)
              (config.INPUT_SEQUENCE_LENGTH-1)*config.DIM +  # velocity history
              1 +                           # particle type
              2*config.DIM)                 # distances to walls

device = torch.device(config.DEVICE)

model = om.OurModel(
    in_dim=feature_dim, 
    latent_dim=config.LATENT_DIM, 
    k=config.NUM_MP_STEPS, 
    connectivity_radius=config.CONNECTIVITY_RADIUS
).to(device)

train_dataset = ParticleDataset()
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.BATCH_SIZE, 
    num_workers=config.NUM_WORKERS, 
    shuffle=True, 
    collate_fn=collate_fn, 
    pin_memory=True, 
    persistent_workers=True
)

print(f'\nYou may expect: {len(train_loader)} iterations per epoch:\n')

# Calculate gamma for exponential decay from 1e-4 to 1e-6 over 20M steps
steps_per_epoch = len(train_loader)
total_steps = config.MAX_EPOCHS * steps_per_epoch
gamma = (1e-6 / 1e-4) ** (1 / total_steps)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
criterion = nn.MSELoss()

# Initialize training state
start_epoch = 0
best_loss = inf
global_step = 0  # Track total number of steps for scheduler

if args.resume_from:
    print(f"Loading checkpoint from {args.resume_from}")
    checkpoint = torch.load(args.resume_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
    
    # Restore scheduler state by stepping to correct point
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    for _ in range(global_step):
        scheduler.step()
        
    print(f"Resuming from epoch {start_epoch} with best loss {best_loss}")
    print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

os.makedirs(config.OUT_DIR, exist_ok=True)

for epoch in range(start_epoch, config.MAX_EPOCHS):
    model.train()
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
        global_step += 1
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
            'global_step': global_step,
        }, config.OUT_DIR / f"best_model.pth")
    
    print(f"\nEpoch [{epoch+1}/{config.MAX_EPOCHS}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    if (epoch + 1) % 1 == 0 or epoch + 1 == config.MAX_EPOCHS:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'global_step': global_step,
        }, config.OUT_DIR / f"model_epoch_{epoch+1}.pth")

print("Training complete.")