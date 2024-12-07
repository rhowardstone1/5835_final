
#Run this first:
# 
#  pip install torch, tensorflow, torch-geometric   # (if needed)
# 
#  git clone https://github.com/rhowardstone1/5835_final.git
#  cd 5835_final
#  ./setup.sh   # (this script)
# 

dataset_name="Water"
wd=pwd

#cleanup
mkdir scripts_import
mv *.py scripts_import/

echo ''
echo 'Cloning OG repo'
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/deepmind-research.git
cd deepmind-research
git sparse-checkout set learning_to_simulate

echo ''
echo "Downloading Dataset ${dataset_name}..."
mkdir -p "${wd}/datasets"
bash ${wd}/deepmind-research/learning_to_simulate/download_dataset.sh "${dataset_name}" "${wd}/datasets"

echo ''
echo "Converting to hdf5"
echo " YOU MAY IGNORE THESE ERRORS... I THINK! "
echo ''
python scripts_import/Convert_TFRecord_to_hdf5.py "${wd}/datasets/${dataset_name}" "${wd}/datasets/${dataset_name}_hdf5"

echo ''
echo 'Randomly subsampling training dataset with seed=42'
python scripts_import/Randomly_subsample_dataset.py --seed 42 "${wd}/datasets/${dataset_name}_hdf5/train.h5" "${wd}/datasets/${dataset_name}100_hdf5"

echo ''
echo " NOW GO TO ${wd}/scripts_import/config.py AND ADJUST THE PARAMETERS TO YOUR LIKING "
echo "   YOU WILL ALSO NEED TO ADJUST ALL ' #  data paths'  "
echo ''
echo " THEN, cd to ${wd}/scripts_import/  AND RUN:"
echo "   python train_our_model.py   "
echo ''



