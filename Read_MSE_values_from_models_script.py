

import os, torch


base = '/data/rye/5835/models/'

model_dirs = os.listdir(base)

for model_dir in model_dirs:
	print('\n', model_dir)
	files = os.listdir(base+model_dir)

	for f in files:
		if f.find('epoch')!=-1:
			model = torch.load(base+model_dir+'/'+f)
			print(f, model_dir, model['loss'])


	
	