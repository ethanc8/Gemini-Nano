GIT_LFS_SKIP_SMUDGE=1 git clone https://ethanc8:${HF_TOKEN}@huggingface.co/ethanc8/Gemini-Nano-CI

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
rm Miniforge3-$(uname)-$(uname -m).sh

bash convert-all.sh $1

PATH="$HOME/miniforge3/condabin/:$PATH" mv weights_$1.safetensors Gemini-Nano-CI/weights_$1.safetensors

cd Gemini-Nano-CI
git add weights_$1.safetensors && git commit -m "Upload of $1 weights"
git push
cd ..