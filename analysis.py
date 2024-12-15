import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from data import ParticleDataset
from our_model import OurModel
import config
from data import collate_fn
from tqdm import tqdm

def analyze(model_dir, model_prefix):
	device = torch.device(config.DEVICE)
	
	dataloader = DataLoader(ParticleDataset(split="valid"), 
						  batch_size=config.BATCH_SIZE, 
						  num_workers=config.NUM_WORKERS, 
						  shuffle=True, 
						  collate_fn=collate_fn, 
						  pin_memory=True, 
						  persistent_workers=True)
	
	loss_fn = nn.MSELoss().to(device)
	
	temp = model_dir.split("_")
	for t in temp:
		if t.startswith('mp'):
			message_passing_steps = int(t.replace("mp",""))
		elif t.startswith('sl'):
			sequence_length = int(t.replace("sl",""))
	config.NUM_MP_STEPS = message_passing_steps
	config.INPUT_SEQUENCE_LENGTH = sequence_length
			
	
	
	
	results = []
	
	
	on_num = 0
	# Iterate through model folders 
	for file in sorted([ele for ele in os.listdir(model_dir) if ele.find('epoch')!=-1], key = lambda x:int(x.split('_')[-1].split(".")[0])):
		on_num += 1
		if on_num%5 == 0 or on_num in [1,50]:
			print("On num is: ", on_num)
			if file.startswith(model_prefix) and file.endswith(".pth"):
				feature_dim = (config.DIM + (sequence_length-1)*config.DIM + 1 + 2*config.DIM)

				epoch = int(file.split("_")[-1].split(".")[0])
				print("On epoch:", epoch)
				
				model_path = os.path.join(model_dir, file)

				# Load the model
				model_data = torch.load(model_path, map_location=device)

				model = OurModel(in_dim=feature_dim, 
							   latent_dim=config.LATENT_DIM, 
							   k=message_passing_steps, 
							   connectivity_radius=config.CONNECTIVITY_RADIUS)

				model.load_state_dict(model_data['model_state_dict'])
				model.to(device)  # Move model to GPU
				model.eval()  # Set to evaluation mode

				# Calculate MSE loss on the dataset
				total_loss = 0.0
				total_samples = 0

				with torch.no_grad():
					for batch_idx, (positions, node_features, accelerations, padding_masks) in tqdm(enumerate(dataloader)):
						# Move all tensors to GPU
						positions = positions.to(device)
						node_features = node_features.to(device) 
						accelerations = accelerations.to(device)
						padding_masks = padding_masks.to(device)

						outputs = model(node_features, positions)
						loss = loss_fn(outputs[padding_masks], accelerations[padding_masks])
						total_loss += loss.item() * positions.size(0)
						total_samples += positions.size(0)

				mse_loss = total_loss / total_samples
				print(f"Epoch {epoch}: MSE Loss = {mse_loss}")

				# Record the result
				results.append({"epoch": epoch, "mse_loss": mse_loss})
			
	# Save results to a CSV file
	results_df = pd.DataFrame(results)
	results_df.sort_values("epoch", inplace=True)  # Ensure results are sorted by epoch
	outname = f"/data/rye/5835/evaluation/mse_loss_over_time_{model_dir.replace('/','')}.csv"
	results_df.to_csv(outname, index=False)
	print(f"MSE loss over time recorded in {outname}")

if __name__ == "__main__":
	base = "/data/rye/5835/models"
	
	dirs = os.listdir(base)[1:]
	print(dirs)
	for d in dirs:
		print(f"Beginning {d}")
		analyze(base+'/'+d, "model_epoch_")
		print()