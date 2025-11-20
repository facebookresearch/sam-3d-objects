# Setup

You will need an NVIDIA GPU with **at least 32 GB of VRAM**; the code runs across multiple GPU architectures, including Blackwell.

## 1. Setup Python Environment

`uv` is now the default way to create the environment. The additional NVIDIA/PyTorch indices and Kaolin find-links are configured in `pyproject.toml`, so no extra exports are needed. Use Python 3.11 (the project targets 3.11 only). **CUDA toolkit 12.9 is required for building gsplat/pytorch3d**; install it from the [CUDA 12.9 archive](https://developer.nvidia.com/cuda-12-9-0-download-archive) and export the paths before running `uv sync`:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Adding these exports to `~/.bashrc` is recommended so the correct toolkit is always picked up.

Example workflow:

```bash
# install uv if you don't have it yet
pip install uv

# create and activate a local virtualenv
uv python install 3.11
uv venv .venv
source .venv/bin/activate

# install the base set of pinned dependencies
uv sync
# useful flags
# uv sync --preview-features extra-build-dependencies  # silence warnings for git-based deps needing torch at build time

# optional extras
uv sync --extra dev       # developer tooling
uv sync --extra p3d       # pytorch3d + flash_attn
uv sync --extra inference # kaolin/gsplat/gradio extras
# for demo/inference runs, you typically want both p3d and inference extras installed

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

> If you still prefer a Conda-based workflow for GPU toolchains, you can reuse `environments/default.yml` to provision system libraries, then activate your environment and run `uv sync` inside it to install Python dependencies.

## 2. Getting Checkpoints

### From HuggingFace

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects
Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

⚠️ SAM 3D Objects is available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions.
Sanctioned jurisdiction will result in requests being **rejected**.

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```
