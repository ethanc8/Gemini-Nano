# Open-source Gemini Nano inference

The goal of this project is to be able to infer Gemini models, such as Gemini Nano, 
with support for multimodal inference (image, audio, video, text input and image/text output).

## Download weights

The weights of Gemini Nano, extracted from Chrome Canary 128.0.6557.0, are available at https://huggingface.co/oongaboongahacker/Gemini-Nano.

## Installation of dependencies

### Installation of Conda

First, please have Conda installed on your computer. If it's not installed, please install [Miniforge3](https://conda-forge.org/miniforge/), which includes Conda and a conda-forge based Python environment. You can install Miniforge3 using the following command:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

Close and reopen your shell, and run:

```bash
# Prevent Conda from polluting your environment when you're not working on Conda-managed projects.
conda config --set auto_activate_base false
```

### CPU-only with Conda

Now, you can use Conda to install the dependencies.

```bash
mamba env create -f environment-cpu.yml
mamba activate Gemini-Nano
```

If you modify `environment.yml`, please run

```bash
mamba env update -f environment-cpu.yml
```
