#!/bin/bash
#SBATCH -p a40             # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:4      # request GPU(s)
#SBATCH -c 32              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=maskdino_lifeplan_b_merged_tiled_t4v2_demo_test # customize this for your project
source ~/.bashrc
source activate md3
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# The "${@}" allows us to pass through arguments at the end of the sbatch command to our script, main.py.
python demo.py --config-file /h/jquinto/MaskDINO/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
--input /h/jquinto/GRLLN6_256_v6/*.png \
--output /h/jquinto/MaskDINO/demo/GRLLN6_merged_v6_pretrained \
--opts MODEL.WEIGHTS /h/jquinto/MaskDINO/output_merged_v6_multiscale/model_0001999.pth
