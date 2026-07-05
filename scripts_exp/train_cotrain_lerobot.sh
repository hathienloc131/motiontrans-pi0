# bash scripts_exp/train_cotrain_lerobot.sh
#
# LeRobot-dataset variant of train_cotrain.sh (see get_normalize_cotrain_lerobot.sh for the
# dataset requirements). Run get_normalize_cotrain_lerobot.sh once first to compute norm
# stats and the train/val split for this repo_id.

exp_name="pp_abs_20hz"                           # the description of the experiment target
repo_id="pp_abs_20hz"          # identifier only, used for norm-stats/assets dir naming
dataset_root="/mnt/data/sftp/data/locht1/vr_data/human_stack_three_cup|/mnt/data/sftp/data/locht1/vr_data/robot_stack_three_cup"   # local directory containing the LeRobot dataset (meta/, data/, videos/)
# Multiple LeRobot datasets (e.g. human + robot co-training): separate the roots with '|',
#   dataset_root="/path/to/human_data|/path/to/robot_data"
# The list and its ORDER must match the get_normalize_cotrain_lerobot.sh run (norm stats and
# the per-dataset train_val_split_{i}.json files are computed over/indexed by this list).
# NOTE: dataset_path is intentionally left unset/empty so data_loader.create_dataset()
# routes through the LeRobot branch (MotionTransDataset) instead of ZarrDataset.

checkpoint_base_dir="/mnt/data/sftp/data/locht1/motiontrans/pretrained_ckpts"
assets_base_dir="/mnt/data/sftp/data/locht1/motiontrans/assets"
export HF_HOME="/mnt/data/sftp/data/locht1/hf"
export OPENPI_DATA_HOME="/mnt/data/sftp/data/locht1/motiontrans/openpi"
export HF_LEROBOT_HOME="/mnt/data/sftp/data/locht1/motiontrans/lerobot"

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

alpha=0.5
proprioception_droprate=0.0
num_devices=2
single_batch_size=24
batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

num_train_steps=80001
keep_period=40000
log_interval=250
save_interval=40000
val_interval=5000
max_token_len=150

single_val_batch_size=24
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed.

# ======== pi0 cocktail  =====
# WANDB_DISABLED=True
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi0_droid_motiontrans \
--exp-name="${now_date}_${now_seconds}_${repo_id}_${exp_name}" \
--alpha=${alpha} \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--batch-size=$batch_size \
--repo_id=${repo_id} \
--dataset_root="${dataset_root}" \
--state_down_sample_steps 0 \
--action_down_sample_steps 1 \
--proprioception_rep "abs" \
--action_rep "relative" \
--proprioception_droprate ${proprioception_droprate} \
--use_val_dataset \
--val_batch_size=$val_batch_size \
--num_train_steps ${num_train_steps} \
--keep_period ${keep_period} \
--log_interval ${log_interval} \
--save_interval ${save_interval} \
--val_interval ${val_interval} \
--model.max_token_len ${max_token_len} \

# --single_arm: add this flag if your dataset only has robot0_*/gripper0_* columns.
# Must match whatever get_normalize_cotrain_lerobot.sh used (state/action dims, norm stats,
# and the train/val split all depend on single_arm being consistent between the two runs).
