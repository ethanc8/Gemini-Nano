#!/bin/bash

mamba env create -f environment-cpu.yml
mamba activate Gemini-Nano
echo "Downloading weights..."
wget https://huggingface.co/oongaboongahacker/Gemini-Nano/resolve/main/weights.bin --no-verbose

echo "Converting weights..."
cd playground
python3 converter.py ../weights.bin ../weights_$1.safetensors
cd ..