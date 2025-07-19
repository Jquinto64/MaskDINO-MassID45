import os
import pickle
import pandas as pd

BS = 8
N_EPOCHS = 5 # MAY WANT TO CHANGE TO 5
STEPS = (327778/368750, 355092/368750)
FILTER_EMPTY = False

if FILTER_EMPTY:
   num_files = 13622
else:
  num_files = len(os.listdir("/h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_341_ignore_neg/train2017"))
num_iters_per_epoch = num_files/BS
max_iters = num_iters_per_epoch*N_EPOCHS

print(f"Number of Training Images: {num_files}")
print(f"MAX_ITERS: {int(max_iters)}")
print(f"WARMUP_ITERS: {int(0.3*max_iters)}")
print(f"STEPS: ({int(STEPS[0]*max_iters)}, {int(STEPS[1]*max_iters)})")
print(f"CHECKPOINT_PERIOD: {num_iters_per_epoch}")
print(f"EVAL_PERIOD: {num_iters_per_epoch}")


# # Get max detections appropriate for dataset
# with open("/h/jquinto/LIFEPLAN_SAHI_tiling/backup/sahi_256_ignore_neg_backup/train2017_masked.pkl", "rb") as f:
#     postprocessed_annots = pickle.load(f)

# # Get number of annotations per image
# lens = []
# for key in postprocessed_annots.keys():
#   l = len(postprocessed_annots[key])
#   lens.append(l)
# print(pd.Series(lens).describe())
