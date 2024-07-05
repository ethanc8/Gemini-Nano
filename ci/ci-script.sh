echo "Cloning https://huggingface.co/ethanc8/Gemini-Nano-CI..."
GIT_LFS_SKIP_SMUDGE=1 git clone https://ethanc8:${HF_TOKEN}@huggingface.co/ethanc8/Gemini-Nano-CI

echo "Installing Miniforge3..."
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" --no-verbose
bash Miniforge3-$(uname)-$(uname -m).sh -b
rm Miniforge3-$(uname)-$(uname -m).sh

source ~/.bashrc

mamba init
bash convert-all.sh $1

echo "Moving weights to Gemini-Nano-CI..."
mv weights_$1.safetensors Gemini-Nano-CI/weights_$1.safetensors

cd Gemini-Nano-CI
echo "Commiting and pushing weights to Gemini-Nano-CI..."
git add weights_$1.safetensors && git commit -m "Upload of $1 weights"
git push
cd ..