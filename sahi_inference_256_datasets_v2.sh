#!/bin/bash
#SBATCH --gres=gpu:l40s:1   # request GPU(s)
#SBATCH --cpus-per-task=8   # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=sahi_inference_512_sahi_tiled_v9_025_conf_final_R50_val_set_results_0.6_overlap_5_epochs_bs_8_lr_5e-5_one_cycle_colour_augs_15k_iters_reproduce

ENV_NAME=maskdino
module load StdEnv/2020 gcc/9.3.0
module load python/3.10.2
module load cuda/11.8.0
module load opencv/4.8.0
source /home/jquinto/projects/aip-gwtaylor/jquinto/virtualenvs/$ENV_NAME/bin/activate

# Debugging outputs
pwd
python --version
pip freeze

#################################### VARIABLE TILE_SIZE (15k iters) #############################################

TILE_SIZE=512
python sahi_inference.py --model_path /h/jquinto/MaskDINO/output_lifeplan_b_${TILE_SIZE}_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters/model_final.pth \
--exp_name lifeplan_b_${TILE_SIZE}_sahi_tiled_keep_cutoff_v9_R50_one_cycle_5e-5_epoch_4_15k_iters_reprod_maskdino \
--dataset_json_path /h/jquinto/lifeplan_b_v9_cropped_center/annotations/instances_val2017.json \
--dataset_img_path /h/jquinto/lifeplan_b_v9_cropped_center/val2017 \
--config_path /h/jquinto/MaskDINO/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
--crop_fac 16 \
--postprocess_match_threshold 0.5 \
--model_confidence_threshold 0.25 \
--predict \
--scale_factor 1 \
--slice_height ${TILE_SIZE} \
--slice_width ${TILE_SIZE} \
--overlap 0.6

