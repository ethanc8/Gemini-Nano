conda activate Gemini-Nano
echo "PATH: $PATH"

echo "Cloning https://huggingface.co/ethanc8/Gemini-Nano-CI..."
GIT_LFS_SKIP_SMUDGE=1 git clone https://ethanc8:${HF_TOKEN}@huggingface.co/ethanc8/Gemini-Nano-CI

bash convert-all.sh $1

echo "Moving weights to Gemini-Nano-CI..."
mv weights_$1.safetensors Gemini-Nano-CI/weights_$1.safetensors

cd Gemini-Nano-CI
echo "Commiting and pushing weights to Gemini-Nano-CI..."
git add weights_$1.safetensors && git commit -m "Upload of $1 weights"
git push
cd ..