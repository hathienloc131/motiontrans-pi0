# bash scripts_exp/train_cotrain_lerobot.sh
#
# LeRobot-dataset variant of train_cotrain.sh (see get_normalize_cotrain_lerobot.sh for the
# dataset requirements). Run get_normalize_cotrain_lerobot.sh once first to compute norm
# stats and the train/val split for this repo_id.

exp_name="default"                           # the description of the experiment target
repo_id="Put YOUR_LEROBOT_REPO_ID Here"      # identifier only, also used for norm-stat/assets dir naming
dataset_root="Put /path/to/your/lerobot_dataset Here"   # local directory containing the LeRobot dataset (meta/, data/, videos/)
# NOTE: dataset_path is intentionally left unset/empty so data_loader.create_dataset()
# routes through the LeRobot branch (MotionTransDataset) instead of ZarrDataset.

checkpoint_base_dir="/mnt/data/sftp/data/locht1/motiontrans/pretrained_ckpts"
assets_base_dir="/mnt/data/sftp/data/locht1/motiontrans/assets"
export HF_HOME="/mnt/data/sftp/data/locht1/hf"
export OPENPI_DATA_HOME="/mnt/data/sftp/data/locht1/motiontrans/openpi"
export LEROBOT_HOME="/mnt/data/sftp/data/locht1/motiontrans/lerobot"

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

alpha=0.5
proprioception_droprate=0.0
num_devices=8
single_batch_size=24
batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

num_train_steps=150001
keep_period=75000
log_interval=250
save_interval=75000
val_interval=5000
max_token_len=150

single_val_batch_size=24
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed.

# ======== pi0 cocktail  =====
# WANDB_DISABLED=True
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_droid_motiontrans \
--exp-name="${now_date}_${now_seconds}_${repo_id}_${exp_name}" \
--alpha=${alpha} \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--batch-size=$batch_size \
--repo_id=${repo_id} \
--dataset_root=${dataset_root} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
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
