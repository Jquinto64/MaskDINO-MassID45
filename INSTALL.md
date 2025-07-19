## Installation

### MassID45 setup script
```bash
conda create --name maskdino python=3.8 -y
conda activate maskdino
# optional: module load cuda-11.3 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U opencv-python==4.8.1.78

# Navigate to MaskDINO directory
cd MaskDINO

# Install prebuilt detectron2 - see https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Move the following files to the location of your conda environments (.e.g, miniconda/envs/maskdino/lib/python3.8/site-packages/detectron2/)
cp detectron2_modifications/defaults.py /miniconda/envs/maskdino/lib/python3.8/site-packages/detectron2/engine/defaults.py

cp detectron2_modifications/augmentation_impl.py /miniconda/envs/maskdino/lib/python3.8/site-packages/detectron2/data/transforms/augmentation_impl.py

# Install base requirements
pip install -r requirements.txt

# Install certain versions of packages to avoid errors later on
pip install numpy==1.23.1
pip install pillow==9.5.0
pip install sahi==0.11.18
pip install pycocotools

# Move the following files to miniconda/envs/maskdino/lib/python3.8/site-packages/sahi/:
# Change line 17 in sahi_modifications/detectron2.py: sys.path.insert(0, <ABSOLUTE FILE PATH TO MaskDINO folder>) then
cp sahi_modifications/detectron2.py /miniconda/envs/maskdino/lib/python3.8/site-packages/sahi/models/detectron2.py

cp sahi_modifications/annotation.py /miniconda/envs/maskdino/lib/python3.8/site-packages/sahi/annotation.py

# Build pixel decoder dependencies (may take a LONG time)
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```


### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit. If running on the Vector cluster, first run `module load cuda-11.3`.

```bash
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```


