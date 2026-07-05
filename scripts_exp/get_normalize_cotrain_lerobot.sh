repo_id="pp_abs_20hz"          # identifier only, used for norm-stats/assets dir naming
dataset_root="/mnt/data/sftp/data/locht1/vr_data/human_stack_three_cup|/mnt/data/sftp/data/locht1/vr_data/robot_stack_three_cup"   # local directory containing the LeRobot dataset (meta/, data/, videos/)
# Multiple LeRobot datasets (e.g. human + robot co-training): separate the roots with '|',
#   dataset_root="/path/to/human_data|/path/to/robot_data"
# Norm stats and the train/val splits are then computed over the merged datasets; use the
# exact same list/order in train_cotrain_lerobot.sh (splits are saved per dataset as
# train_val_split_{i}.json, indexed by position in the list).
# NOTE: dataset_path is intentionally left unset/empty so data_loader.create_dataset()
# routes through the LeRobot branch (MotionTransDataset) instead of ZarrDataset.

checkpoint_base_dir="/mnt/data/sftp/data/locht1/motiontrans/pretrained_ckpts"
assets_base_dir="/mnt/data/sftp/data/locht1/motiontrans/assets"
export HF_HOME="/mnt/data/sftp/data/locht1/hf"
export OPENPI_DATA_HOME="/mnt/data/sftp/data/locht1/motiontrans/openpi"
export HF_LEROBOT_HOME="/mnt/data/sftp/data/locht1/motiontrans/lerobot"
export CUDA_VISIBLE_DEVICES=0
exp_name="pp_abs_20hz"               # not used

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed.

python scripts/compute_norm_stats.py pi0_droid_motiontrans \
--exp_name=${exp_name} \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--repo_id=${repo_id} \
--dataset_root="${dataset_root}" \
--state_down_sample_steps 0 \
--action_down_sample_steps 1 \
--proprioception_rep "abs" \
--action_rep "relative" \
--use_val_dataset --create_train_val_split --val_ratio=0.025 \
--compute_norm_stats \
--no_wandb_enabled \

# --single_arm: pass this flag if your dataset only has robot0_*/gripper0_* columns (single-arm).
# Omit it (as above) for bimanual data with robot0_*+robot1_* columns present on every row;
# MotionTransDataset assumes a homogeneous schema and does not support per-episode missing arms.

# image_down_sample_steps / state_down_sample_steps: list, get idx - steps[i] for i in down_sample_steps.
# action_down_sample_steps: int, get idx + i * action_down_sample_steps for i in range(action_horizon)