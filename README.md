## Installation
#### OPTION 1: If on the Killarney (Compute Canada) or similar cluster with prebuilt wheels 
```bash
module load StdEnv/2020  gcc/9.3.0
module load python/3.10.2
module load cuda/11.8.0
module load opencv/4.8.0

cd projects/aip-gwtaylor/jquinto
ENV_NAME=maskdino
virtualenv --no-download virtualenvs/$ENV_NAME
source /home/jquinto/projects/aip-gwtaylor/jquinto/virtualenvs/$ENV_NAME/bin/activate

python -m pip show pip
pip install --no-index --upgrade pip
pip install --no-index detectron2
pip install torchvision==0.14.1 torchaudio==0.13.1
pip install --no-index opencv

pip install numpy==1.23.1
pip install sahi==0.11.18
pip install pycocotools

pip install --no-index  cython
pip install --no-index  scipy
pip install --no-index  shapely
pip install --no-index  timm
pip install --no-index  h5py
pip install --no-index  submitit
pip install --no-index  scikit-image

# Run interactive jobs with something like this:
srun --account=your_aip_project --gres=gpu:l40s:1 --mem=8G -c 4 --time=2:00:00 --pty bash
source /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/bin/activate
cd maskdino/modeling/pixel_decoder/ops
sh make.sh

# CHANGE LINE 46 in /home/jquinto/projects/aip-gwtaylor/jquinto/virtualenvs/md5/lib/python3.10/site-packages/detectron2/data/transforms/transform.py to:
#    def __init__(self, src_rect, output_size, interp=Image.BILINEAR, fill=0):

# Move the following files to the location of your conda environments
cp detectron2_modifications/defaults.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/detectron2/engine/defaults.py

cp detectron2_modifications/augmentation_impl.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/detectron2/data/transforms/augmentation_impl.py

# Move the following files to miniconda/envs/maskdino/lib/python3.10/site-packages/sahi/:
# Change line 17 in sahi_modifications/detectron2.py: sys.path.insert(0, <ABSOLUTE FILE PATH TO MaskDINO folder>) then
cp sahi_modifications/detectron2.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/sahi/models/detectron2.py

cp sahi_modifications/annotation.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.8/site-packages/sahi/annotation.py
```

#### OPTION 2: If your cluster supports arbitrary package versions
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

## Instructions for Inference on your Images
1.  First download the MS-COCO pretrained R50 checkpoint and place it in this folder:
```bash
wget https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth
``` 
2. Download the model checkpoints from [Zenodo](https://zenodo.org/records/15479862/files/model_checkpoints.zip?download=1). This submodule is for Mask DINO so use `model_final_maskdino.pth`.
3. Run `standalone_inference_maskdino.py` as follows:
```bash
python standalone_inference_maskdino.py --model_path path/to/model_final_maskdino.pth
--imgs_directory path/to/your_images --output_dir path/to/your_output_folder --config configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml
``` 
## Training and Inference Instructions
1. First download the MS-COCO pretrained R50 checkpoint using
```bash
wget https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth
```

2. Ensure your dataset is structured according to the COCO dataset format:
```
your_dataset/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
``` 

3. Make any desired modifications to the data augmentations with `maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py`, and modify the configs `configs/lifeplan/instance-segmentation/Base-Lifeplan-InstanceSegmentation.yaml` and `configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml` as needed (e.g., change NUM_CLASSES based on your desired number of classes).

4. Modify the  `--dataset_path` and `MODEL.WEIGHTS` arguments in `train.sh` to reflect the locations of the MassID45 training data and pretrained model weights from Step 1. Then run the training script with 
```bash
sbatch train.sh
```
Outputs will be saved in the folder specified by `OUTPUT_DIR`.

5.  Replace `--dataset_img_path` and `--dataset_json_path` in `sahi_inference_256_datasets_v2.sh` with the locations of the validation or testing data, respectively, then run inference with: 
```bash
sbatch sahi_inference_256_datasets_v2.sh
```
Results will appear in the `runs/predict` folder under the name specified by the `--exp_name` argument.
