from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import dataclasses
import os
import json
import torch
import copy
import scipy.interpolate as si
import scipy.spatial.transform as st
import random
from openpi.policies.pose_util import pose_to_mat, mat_to_pose10d, mat_to_pose
from openpi.policies.pose_repr_util import convert_pose_mat_rep


def _obs_col(name: str) -> str:
    """Maps a short internal low-dim key (e.g. "robot0_eef_pos") to the LeRobot dataset's
    prefixed column name (e.g. "observation.robot0_eef_pos"), following the standard LeRobot
    convention of namespacing all observation fields under "observation." (only "action" and
    the standard per-row metadata fields stay unprefixed)."""
    return f"observation.{name}"


class MotionTransDataset(LeRobotDataset):
    def __init__(self, data_config, action_horizon: int, split_filename: str = 'train_val_split.json'):
        super().__init__(data_config.repo_id, root=data_config.dataset_root)
        self.data_config = data_config
        # name of the train/val split file under norm_stats_dir; MultiMotionTransDataset gives
        # each sub-dataset its own file so splits don't collide in the shared assets dir
        self.split_filename = split_filename
        self.alpha = data_config.alpha
        self.single_arm = data_config.single_arm
        self.image_hisory_length = len(data_config.image_down_sample_steps) + 1
        self.image_down_sample_steps = data_config.image_down_sample_steps
        self.state_hisory_length = len(data_config.state_down_sample_steps) + 1
        self.state_down_sample_steps = data_config.state_down_sample_steps
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps
        self.compute_norm_stats = data_config.compute_norm_stats
        self.proprioception_rep = data_config.proprioception_rep
        self.action_rep = data_config.action_rep
        self.proprioception_droprate = data_config.proprioception_droprate

        # NOTE: assumes a homogeneous LeRobot schema where every row has robot1_* columns
        # populated whenever single_arm=False (no per-episode missing-arm handling, unlike
        # ZarrDataset which supports mixed single-/dual-arm episodes via zero-filling).
        self.n_robots = 1 if self.single_arm else 2
        self.low_dim_keys = []
        for robot_idx in range(self.n_robots):
            self.low_dim_keys += [
                f"robot{robot_idx}_eef_pos",
                f"robot{robot_idx}_eef_rot_axis_angle",
                f"gripper{robot_idx}_gripper_pose",
            ]
        self.low_dim_features = {}
        for key in self.low_dim_keys:
            self.low_dim_features[key] = torch.stack(self.hf_dataset[_obs_col(key)]).numpy().astype(np.float32)
        self.is_human_list = torch.stack(self.hf_dataset[_obs_col('is_human')]).numpy().astype(np.float32)
        n_human = int(self.is_human_list.sum())
        n_robot = len(self.is_human_list) - n_human
        if n_human == 0:
            self.alpha = 1.0
        elif n_robot == 0:
            self.alpha = 0.0
        else:
            alpha_robot = self.alpha / n_robot
            alpha_human = (1 - self.alpha) / n_human
            adjust_alpha = alpha_robot / (alpha_robot + alpha_human)
            self.alpha = adjust_alpha
        self.actions = torch.stack(self.hf_dataset['action']).numpy().astype(np.float32)
        # Raw `action` column is packed as n_robots equal-sized chunks: first 6 dims of each
        # chunk are the eef pose (pos + rotvec), the remainder is that arm's gripper command.
        assert self.actions.shape[-1] % self.n_robots == 0, (
            f"action dim {self.actions.shape[-1]} not divisible by n_robots={self.n_robots}"
        )
        self.action_dim_per_robot = self.actions.shape[-1] // self.n_robots
        # camera0_pose is optional, matching ZarrDataset: episodes with a static (non-moving)
        # camera (e.g. robot teleop data) may not have this column at all, in which case
        # cam_proj falls back to identity in __getitem__ instead of crashing.
        if _obs_col('camera0_pose') in self.hf_dataset.column_names:
            self.camera_poses = torch.stack(self.hf_dataset[_obs_col('camera0_pose')]).numpy().astype(np.float32)
        else:
            self.camera_poses = None

        if data_config.create_train_val_split:
            assert data_config.use_val_dataset
        if data_config.create_train_val_split:
            self.create_train_val_split()
        if data_config.use_val_dataset:
            self.indices = self.get_indices('train')
        else:
            self.indices = list(range(len(self.hf_dataset)))


    def create_train_val_split(self):
        episode_num = len(self.hf_dataset['timestamp'])
        val_num = int(episode_num * self.data_config.val_ratio)
        np.random.seed(self.data_config.seed)
        train_episode_idx = np.random.choice(episode_num, episode_num - val_num, replace=False)
        train_episode_idx = np.sort(train_episode_idx)
        val_episode_idx = np.setdiff1d(np.arange(episode_num), train_episode_idx)
        os.makedirs(self.data_config.norm_stats_dir, exist_ok=True)
        with open(os.path.join(self.data_config.norm_stats_dir, self.split_filename), 'w') as f:
            json.dump({'train_episode_idx': train_episode_idx.tolist(), 'val_episode_idx': val_episode_idx.tolist()}, f)


    def get_indices(self, split):
        with open(os.path.join(self.data_config.norm_stats_dir, self.split_filename), 'r') as f:
            split_idx = json.load(f)[f'{split}_episode_idx']
        indices = [idx for idx in split_idx]
        return indices
    
    
    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.get_indices('val')
        return val_set
    
    
    def set_sample_ratio(self, sample_ratio):
        interval_size = int(1.0 / sample_ratio)
        self.indices = self.indices[::interval_size]


    def __len__(self):
        return len(self.indices)
    
    
    def get_prob(self, start_step, end_step, now_step, start_prob=0.8, end_prob=0.4):
        # from start_prob -> end_prob linearly
        assert start_step <= now_step < end_step
        return start_prob - (start_prob - end_prob) * (now_step - start_step) / (end_step - start_step)
    

    def __getitem__(self, idx) -> dict:
        """
            return MotionTrans data item
            Dict:
                - image_1: image at time t
                - image_2: image at time t-1
                ...
                - image_n: image at time t-n+1
                - state: state at time t ~ t-n+1, dim: n*(3+6+6) = n*15
                - action_is_pad: action is padding or not, dim: n
                - actions: action at time t ~ t+n-1, dim: (n, (3+6+6)) = (n, 15)
        """
        idx = self.indices[idx]
        low_dim_dict, return_dict = {}, {}
        current_idx_item = self.hf_dataset[idx]
        episode_index = self.hf_dataset[idx]['episode_index'].item()
        start_idx = self.episode_data_index["from"][episode_index].item()
        end_idx = self.episode_data_index["to"][episode_index].item()
        if self.camera_poses is not None:
            cam_proj = np.linalg.inv(pose_to_mat(self.camera_poses[idx])) @ pose_to_mat(self.camera_poses[start_idx])
        else:
            cam_proj = np.eye(4)

        # get image and image history
        image_target_idx = np.array([idx] + [idx - self.image_down_sample_steps[history_idx] for history_idx in range(self.image_hisory_length - 1)])
        image_target_idx = np.clip(image_target_idx[::-1], start_idx, end_idx - 1)
        image_key = 'observation.images.camera0'
        # NOTE: following the normalization method of src/openpi/models/model.py line 117
        if self.compute_norm_stats:
            # norm stats only cover state/actions; emit placeholder images instead of paying
            # for video decoding (MotionTransInputs still requires the image_* keys to exist)
            for i in range(self.image_hisory_length):
                return_dict['image_{}'.format(i + 1)] = torch.zeros(3, 32, 32)
        elif image_key in self.meta.video_keys:
            # video-backed camera: frames live in mp4 files, not in hf_dataset columns
            query_ts = torch.stack(self.hf_dataset.select([int(i) for i in image_target_idx])['timestamp']).tolist()
            frames = self._query_videos({image_key: query_ts}, episode_index)[image_key]
            if frames.dim() == 3:  # _query_videos squeezes single-frame queries to (C, H, W)
                frames = frames.unsqueeze(0)
            for i in range(self.image_hisory_length):
                return_dict['image_{}'.format(i + 1)] = frames[i] * 2.0 - 1.0
        else:
            for i in range(self.image_hisory_length):
                return_dict['image_{}'.format(i + 1)] = self.hf_dataset[int(image_target_idx[i])][image_key] * 2.0 - 1.0
        
        # print(type(return_dict['image_{}'.format(i + 1)]))
        # print(return_dict['image_{}'.format(i + 1)].dtype)
        # print(return_dict['image_{}'.format(i + 1)].max(), return_dict['image_{}'.format(i + 1)].min())
        
        # get state features and state history
        state_target_idx = np.array([idx] + [idx - self.state_down_sample_steps[history_idx] for history_idx in range(self.state_hisory_length - 1)])
        state_target_idx = np.clip(state_target_idx[::-1], start_idx, end_idx - 1)
        interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
        interpolation_end = min(int(state_target_idx[-1]) + 2 + 5, end_idx)
        for key in self.low_dim_keys:
            input_arr = self.low_dim_features[key]
            if 'eef_pos' in key:
                pair_key = key.replace('eef_pos', 'eef_rot_axis_angle')
                pair_input_arr = self.low_dim_features[pair_key]
                rot_preprocess = st.Rotation.from_rotvec
                rot_postprocess = st.Rotation.as_rotvec
                slerp = st.Slerp(
                    times=np.arange(interpolation_start, interpolation_end),
                    rotations=rot_preprocess(pair_input_arr[interpolation_start: interpolation_end]))
                output_rot = rot_postprocess(slerp(state_target_idx))
                interp = si.interp1d(
                    x=np.arange(interpolation_start, interpolation_end),
                    y=input_arr[interpolation_start: interpolation_end],
                    axis=0, assume_sorted=True)
                output_pos = interp(state_target_idx)
                # need back projection since camera pose may change
                output = np.concatenate([output_pos, output_rot], axis=-1)
                output = pose_to_mat(output)
                output = cam_proj @ output
                output = mat_to_pose(output)
                low_dim_dict[key] = output[:, :3]
                low_dim_dict[pair_key] = output[:, 3:]
            elif 'rot' in key:
                continue
            else:
                interp = si.interp1d(
                    x=np.arange(interpolation_start, interpolation_end),
                    y=input_arr[interpolation_start: interpolation_end],
                    axis=0, assume_sorted=True)
                output = interp(state_target_idx)
                low_dim_dict[key] = output

        # get action chunk
        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        actions = self.actions[idx: slice_end: self.action_down_sample_steps].copy()   # NOTE: need .copy() since self.actions is a class-view !!!!
        # need back projection since camera pose may change (per-robot pose sub-block within
        # the packed action array)
        for robot_idx in range(self.n_robots):
            col_start = robot_idx * self.action_dim_per_robot
            cam_action_pose = pose_to_mat(actions[:, col_start:col_start + 6])
            cam_action_pose = cam_proj @ cam_action_pose
            actions[:, col_start:col_start + 6] = mat_to_pose(cam_action_pose)
        action_is_pad = torch.tensor([False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0]))
        return_dict['action_is_pad'] = action_is_pad
        padding = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
        actions = np.concatenate([actions, padding], axis=0)

        # calculate relative pose to previous pose for obs and action, per robot arm
        final_action_parts = []
        state_parts = []
        for robot_idx in range(self.n_robots):
            pos_key = f"robot{robot_idx}_eef_pos"
            rot_key = f"robot{robot_idx}_eef_rot_axis_angle"
            gripper_key = f"gripper{robot_idx}_gripper_pose"
            col_start = robot_idx * self.action_dim_per_robot

            pose_mat = pose_to_mat(np.concatenate([low_dim_dict[pos_key], low_dim_dict[rot_key]], axis=-1))
            action_mat = pose_to_mat(actions[..., col_start:col_start + 6])
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.proprioception_rep,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.action_rep,
                backward=False)

            # for robot eef proprioception, we ignore identity relative action for eef-pose.
            obs_pose_mat = obs_pose_mat[:-1]
            # for hand / gripper proprioception, we ignore the earliest propriception to make the number of timestamps for eef & gripper the same.
            gripper_obs = low_dim_dict[gripper_key][1:]
            # The final dimension of state should be:  ((3 + 6) + 6) * (state_horizon - 1).
            # If we set state_horizon = 2, then the dimension of state will be (3 + 6 + 6) * (3 - 1) = 30

            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
            action_gripper = actions[..., col_start + 6:col_start + self.action_dim_per_robot]
            final_action_parts.append(np.concatenate([action_pose, action_gripper], axis=-1))
            state_parts.append(np.concatenate([obs_pose.flatten(), gripper_obs.flatten()], axis=-1))

            low_dim_dict[pos_key] = obs_pose[:, :3]
            low_dim_dict[rot_key] = obs_pose[:, 3:]

        final_action = np.concatenate(final_action_parts, axis=-1)
        return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))
        return_dict['state'] = torch.from_numpy(np.concatenate(state_parts, axis=-1).astype(np.float32))
        copy_key = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
        for key in copy_key:
            return_dict[key] = current_idx_item[key]
        drop_val = random.random()
        if drop_val <= self.proprioception_droprate:
            return_dict['state'] = torch.zeros_like(return_dict['state'], dtype=return_dict['state'].dtype)
        
        is_human = self.is_human_list[idx]
        if is_human:
            return_dict['alpha'] = 1 - self.alpha
        else:
            return_dict['alpha'] = self.alpha

        return return_dict


class MultiMotionTransDataset:
    """Concatenation of several MotionTransDataset instances, one per '|'-separated entry in
    data_config.dataset_root (same multi-folder syntax as ZarrDataset). Presents the same
    interface as a single MotionTransDataset (len/getitem/get_val_dataset/set_sample_ratio),
    so norm stats, train/val splits and training all run over the merged data.

    The dataset_root order must be identical between the compute_norm_stats run and the
    training run: each sub-dataset stores its train/val split as train_val_split_{i}.json
    (indexed by position) in the shared norm_stats_dir.
    """

    def __init__(self, data_config, action_horizon: int):
        roots = [root for root in data_config.dataset_root.split('|') if root]
        assert len(roots) > 1, "MultiMotionTransDataset expects multiple '|'-separated dataset roots"
        dataset_class = getattr(data_config, 'dataset_class', MotionTransDataset)
        self.datasets = [
            dataset_class(
                dataclasses.replace(data_config, dataset_root=root),
                action_horizon,
                split_filename=f'train_val_split_{i}.json',
            )
            for i, root in enumerate(roots)
        ]

        # Each sub-dataset computed its human/robot loss-rebalancing alpha from its own frame
        # counts; recompute it over the merged counts and overwrite, so the human/robot
        # gradient weighting matches the configured alpha globally (as ZarrDataset does).
        is_human_all = np.concatenate([d.is_human_list for d in self.datasets])
        n_human = int(is_human_all.sum())
        n_robot = len(is_human_all) - n_human
        alpha = data_config.alpha
        if n_human == 0:
            alpha = 1.0
        elif n_robot == 0:
            alpha = 0.0
        else:
            alpha_robot = alpha / n_robot
            alpha_human = (1 - alpha) / n_human
            alpha = alpha_robot / (alpha_robot + alpha_human)
        for d in self.datasets:
            d.alpha = alpha

        # task_index values are only unique within one LeRobot dataset, so build a merged task
        # table and shift each sub-dataset's task_index by its offset in __getitem__.
        self.task_offsets = []
        self.tasks = {}
        offset = 0
        for d in self.datasets:
            self.task_offsets.append(offset)
            for task_idx, task in d.meta.tasks.items():
                self.tasks[offset + int(task_idx)] = task
            offset += len(d.meta.tasks)

        self._update_sizes()

    def _update_sizes(self):
        self.cumulative_sizes = np.cumsum([len(d) for d in self.datasets])

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, idx) -> dict:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side='right'))
        local_idx = idx - (int(self.cumulative_sizes[dataset_idx - 1]) if dataset_idx > 0 else 0)
        item = self.datasets[dataset_idx][local_idx]
        item['task_index'] = item['task_index'] + self.task_offsets[dataset_idx]
        return item

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.datasets = [d.get_val_dataset() for d in self.datasets]
        val_set._update_sizes()
        return val_set

    def set_sample_ratio(self, sample_ratio):
        for d in self.datasets:
            d.set_sample_ratio(sample_ratio)
        self._update_sizes()