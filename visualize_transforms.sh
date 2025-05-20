#!/bin/bash
#SBATCH -p rtx6000             # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:1    # request GPU(s)
#SBATCH -c 8              # number of CPU cores
#SBATCH --mem=64G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m
#SBATCH --time=5:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=maskdino_sahi_512_tiled_v9_visualization_filter_empty_rand_crop # customize this for your project
source ~/.bashrc
source activate md3
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# Note: the desired dataset for visualization must have the symbolic link: /h/jquinto/MaskDINO/datasets/lifeplan
python visualize_transforms.py --source dataloader \
--config-file  /h/jquinto/MaskDINO/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
--output-dir /h/jquinto/MaskDINO/sample_dataloaders/sample_dataloader_sahi_2x_zoom_v9_1024_keep_cutoff_sr_swinir_bioscan