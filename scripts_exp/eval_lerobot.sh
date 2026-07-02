# bash scripts_exp/eval_lerobot.sh
#
# LeRobot-dataset variant of eval.sh (see get_normalize_cotrain_lerobot.sh for the dataset
# requirements). Uses the same repo_id/train-val-split as train_cotrain_lerobot.sh.

exp_name="0712_pi0_eval_lrsc"                          # the description of the experiment target
repo_id="Put YOUR_LEROBOT_REPO_ID Here"
dataset_root="Put /path/to/your/lerobot_dataset Here"   # local directory containing the LeRobot dataset (meta/, data/, videos/)
policy_dir="Put YOUR_CKPT_PATH Here"
max_token_len=100
# NOTE: dataset_path is intentionally left unset/empty so data_loader.create_dataset()
# routes through the LeRobot branch (MotionTransDataset) instead of ZarrDataset.

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
num_devices=1
single_val_batch_size=24
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

checkpoint_base_dir="checkpoints_pi0/pretrained_ckpts"
assets_base_dir="checkpoints_pi0/assets"
export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export OPENPI_DATA_HOME="checkpoints_pi0/openpi"
export LEROBOT_HOME="checkpoints_pi0/lerobot"
# export WANDB_BASE_URL=https://api.bandw.top

uv run scripts/eval.py pi0_droid_motiontrans \
--exp-name=${exp_name} \
--checkpoint_base_dir=${checkpoint_base_dir} \
--policy_dir=${policy_dir} \
--assets_base_dir=${assets_base_dir} \
--repo_id=${repo_id} \
--dataset_root=${dataset_root} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--use_val_dataset \
--val_batch_size=$val_batch_size \
--model.max_token_len ${max_token_len} \

# --single_arm: add this flag if your dataset only has robot0_*/gripper0_* columns; must
# match whatever train_cotrain_lerobot.sh / get_normalize_cotrain_lerobot.sh used.
