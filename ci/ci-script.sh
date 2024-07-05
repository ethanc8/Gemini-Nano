echo "Cloning https://huggingface.co/ethanc8/Gemini-Nano-CI..."
GIT_LFS_SKIP_SMUDGE=1 git clone https://ethanc8:${HF_TOKEN}@huggingface.co/ethanc8/Gemini-Nano-CI

echo "Installing Miniforge3..."
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" --no-verbose
bash Miniforge3-$(uname)-$(uname -m).sh -b
rm Miniforge3-$(uname)-$(uname -m).sh

__conda_setup="$('$HOME/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "$HOME/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/mamba.sh"
fi

bash convert-all.sh $1

echo "Moving weights to Gemini-Nano-CI..."
mv weights_$1.safetensors Gemini-Nano-CI/weights_$1.safetensors

cd Gemini-Nano-CI
echo "Commiting and pushing weights to Gemini-Nano-CI..."
git add weights_$1.safetensors && git commit -m "Upload of $1 weights"
git push
cd ..