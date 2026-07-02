# bash scripts_exp/get_normalize_cotrain_lerobot.sh
#
# LeRobot-dataset variant of get_normalize_cotrain.sh: instead of reading pre-processed
# .zarr shards (dataset_path), this loads a LeRobot-format dataset via MotionTransDataset
# (src/openpi/policies/dataset_motiontrans.py), read from a local directory (dataset_root)
# rather than repo_id/HF Hub resolution. The dataset must expose the columns
# MotionTransDataset expects: robot0_eef_pos, robot0_eef_rot_axis_angle,
# gripper0_gripper_pose, action, camera0_pose, image, is_human (plus robot1_*/gripper1_*
# for every row when running bimanual, i.e. single_arm unset below).
#
# repo_id is still required (used as the norm-stats/assets identifier), but the actual data
# is read from dataset_root instead of being resolved under $LEROBOT_HOME/<repo_id> or the HF Hub.

repo_id="Put YOUR_LEROBOT_REPO_ID Here"          # identifier only, used for norm-stats/assets dir naming
dataset_root="Put /path/to/your/lerobot_dataset Here"   # local directory containing the LeRobot dataset (meta/, data/, videos/)
# NOTE: dataset_path is intentionally left unset/empty so data_loader.create_dataset()
# routes through the LeRobot branch (MotionTransDataset) instead of ZarrDataset.

checkpoint_base_dir="checkpoints_pi0/pretrained_ckpts"
assets_base_dir="checkpoints_pi0/assets"
export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export OPENPI_DATA_HOME="checkpoints_pi0/openpi"
export LEROBOT_HOME="checkpoints_pi0/lerobot"
export CUDA_VISIBLE_DEVICES=0
exp_name="default"               # not used

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed.

uv run scripts/compute_norm_stats.py pi0_droid_motiontrans \
--exp_name=${exp_name} \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--repo_id=${repo_id} \
--dataset_root=${dataset_root} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--use_val_dataset --create_train_val_split --val_ratio=0.025 \
--compute_norm_stats \
--no_wandb_enabled \

# --single_arm: pass this flag if your dataset only has robot0_*/gripper0_* columns (single-arm).
# Omit it (as above) for bimanual data with robot0_*+robot1_* columns present on every row;
# MotionTransDataset assumes a homogeneous schema and does not support per-episode missing arms.

# image_down_sample_steps / state_down_sample_steps: list, get idx - steps[i] for i in down_sample_steps.
# action_down_sample_steps: int, get idx + i * action_down_sample_steps for i in range(action_horizon)
