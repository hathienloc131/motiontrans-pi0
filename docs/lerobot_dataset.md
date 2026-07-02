# Training from a LeRobot dataset (instead of `.zarr`)

The default MotionTrans co-training workflow (`scripts_exp/train_cotrain.sh`, `get_normalize_cotrain.sh`,
`eval.sh`) reads pre-processed `.zarr` shards via `ZarrDataset`
(`src/openpi/policies/dataset_zarr.py`). This document describes the alternative path that reads a
LeRobot-format dataset via `MotionTransDataset` (`src/openpi/policies/dataset_motiontrans.py`), and the
`scripts_exp/*_lerobot.sh` scripts that drive it.

Which backend gets used is decided by `data_loader.create_dataset()`
(`src/openpi/training/data_loader.py`): if `dataset_path` is non-empty, `ZarrDataset` is used; otherwise, if
the data config has a `dataset_class` (true for all `MotionTransTrainConfig`-based configs), the LeRobot
backend is used. The `*_lerobot.sh` scripts simply leave `dataset_path` unset.

## Quick start

```bash
bash scripts_exp/get_normalize_cotrain_lerobot.sh   # compute norm stats + train/val split (run once)
bash scripts_exp/train_cotrain_lerobot.sh
bash scripts_exp/eval_lerobot.sh
```

Fill in, in each script:
- `repo_id`: an identifier string only (used for the norm-stats/assets directory name — it does **not** need
  to resolve to anything on the HF Hub).
- `dataset_root`: local filesystem path to the LeRobot dataset directory (the folder containing `meta/`,
  `data/`, and `videos/` or `images/`). This is passed as `root=` to LeRobot's `LeRobotDataset` /
  `LeRobotDatasetMetadata`, bypassing `HF_LEROBOT_HOME`/HF Hub resolution entirely.
- `--single_arm`: add this flag if your dataset only has one arm (`robot0_*`/`gripper0_*` columns). Omit it
  for bimanual data (`robot0_*` + `robot1_*`). This flag must be identical across
  `get_normalize_cotrain_lerobot.sh`, `train_cotrain_lerobot.sh`, and `eval_lerobot.sh` — it changes the
  state/action dimensionality and the norm stats/train-val split are computed against one specific choice.

## Required LeRobot dataset schema

`MotionTransDataset` expects a standard LeRobot dataset directory (`meta/info.json`, `meta/episodes.jsonl`,
`meta/tasks.jsonl`, `data/*.parquet`, plus `videos/*.mp4` or decoded image storage) with these columns,
following the standard LeRobot convention of namespacing every observation field under `observation.` (only
`action` and the standard per-row metadata fields stay unprefixed) — note this is a *different, retargeted*
schema from the raw/intermediate human-data LeRobot export in the original `motiontrans` repo
(`lerobot_human_data_conversion_batch.py`, which emits `observation.raw_vr_left_hand`/`observation.state` etc.
before any hand-retargeting + relative-pose processing):

| Column | Shape | Notes |
|---|---|---|
| `observation.images.camera0` | `(H, W, C)`, float `[0, 1]` | Current camera frame. Standard LeRobot image/video decoding already returns `[0,1]` floats. |
| `observation.robot{i}_eef_pos` | `(3,)` | End-effector position, for `i` in `0` (`--single_arm`) or `0,1` (bimanual). |
| `observation.robot{i}_eef_rot_axis_angle` | `(3,)` | End-effector orientation as a rotation vector (axis-angle). |
| `observation.gripper{i}_gripper_pose` | `(g_i,)` | Gripper/finger command, arbitrary width `g_i` (e.g. `1` for a parallel gripper, `6` for the Inspire hand). |
| `action` | `(n_robots * (6 + g),)` | **Unprefixed** (standard LeRobot convention keeps `action` top-level). Packed per-robot chunks of `[pos(3), rotvec(3), gripper(g)]`, right arm (`robot0`) first. **`action.shape[-1]` must be evenly divisible by `n_robots`**, and each chunk's gripper width must equal that arm's `observation.gripper{i}_gripper_pose` width. |
| `observation.camera0_pose` | `(6,)` | Pose (pos + rotvec) of the reference camera at this timestep, expressed relative to the camera pose at the start of the episode (see "Egocentric camera convention" below). **Optional** — omit entirely for episodes with a static camera (e.g. robot teleop with a fixed calibrated camera); if the column is absent, camera correction falls back to identity, matching `ZarrDataset`'s behavior for zarr shards without `camera0_pose`. |
| `observation.is_human` | scalar (bool/0-1) | `1` for human-demo rows, `0` for robot rows. Used both to compute the human/robot `alpha` loss-weight rebalancing and, per-row, to pick `alpha` vs `1 - alpha`. |
| `task_index` | int | Standard LeRobot field (unprefixed); combined with `meta/tasks.jsonl` to produce the `prompt` string fed to the model (via `PromptFromLeRobotTask`, since `MotionTransTrainConfig` always sets `prompt_from_task=True`). Unlike `ZarrDataset` (which reads an `instruction` column or parses it from the `+instruction+` zarr folder name), the LeRobot path has **no filename-based instruction fallback** — every task must be registered in `tasks.jsonl`. |
| `episode_index`, `frame_index`, `timestamp`, `index` | — | Standard LeRobot per-row fields (unprefixed), populated automatically by any `LeRobotDataset` export. |

### Egocentric camera convention

`camera0_pose[t]` and `robot{i}_eef_pos/rot[t]` must both be expressed in the same convention used by the
`.zarr` conversion pipeline: pose relative to the camera's own pose at the *start of the episode*
(`camera_pose = vr2camera0 @ head_pose_mat @ inv(calib_quest2camera)` in
`scripts_data/entry/zarr_human_data_conversion_batch.py` of the main `motiontrans` repo), **not** a world/robot-base
frame. `MotionTransDataset` reprojects this into "relative to the current query timestep" internally
(`cam_proj`) before computing the relative pose representation — see the inline comments in
`dataset_motiontrans.py` for the exact math.

## Bimanual (`single_arm=False`)

`MotionTransDataset` supports two arms (`n_robots=2`), reading `observation.robot0_*`/`observation.gripper0_*`
(treated as "right") and `observation.robot1_*`/`observation.gripper1_*` (treated as "left"), with
`state`/`actions` concatenated as `[right_pose, right_gripper, left_pose, left_gripper]` — matching
`ZarrDataset`'s ordering.

**Assumption:** the LeRobot dataset schema is homogeneous — every row has both robots' columns populated.
Unlike `ZarrDataset`, which supports mixing single-arm human episodes into an otherwise-bimanual dataset
(gracefully zero-filling the missing arm per-episode, since each `.zarr` shard can independently omit
`robot1_*`), a single LeRobot parquet schema can't have a column present for some rows and absent for
others. If your data mixes single- and dual-arm episodes, either export separate LeRobot datasets and only
train one `single_arm` setting at a time, or pad the missing arm's columns with zeros yourself before export.

## Known differences vs. `ZarrDataset`

These are the material, verified differences between `MotionTransDataset` and `ZarrDataset` — same overall
math (camera reprojection → relative pose conversion → pose10d encoding), but different input requirements:

1. **Action packing width.** `ZarrDataset` hard-codes exactly 12 raw columns per robot (`12*robot_idx : 12*robot_idx+12`,
   i.e. a fixed 6-pose + 6-gripper layout, an Inspire-hand-shaped assumption baked into the `.zarr` pipeline).
   `MotionTransDataset` instead derives the per-robot width as `action.shape[-1] // n_robots`, so it works with
   any gripper DOF — but this means the two backends are only numerically identical if your LeRobot `action`
   column also happens to use a fixed 12-per-robot layout. If you're re-exporting the *same* underlying motion
   data through both pipelines and expect bit-identical tensors, make sure the LeRobot `action` column matches
   that convention (or note that they'll differ if it doesn't).
2. **Embodiment source.** `ZarrDataset` infers human-vs-robot from the dataset *folder name* (`'robot' in
   dataset_name`); `MotionTransDataset` requires an explicit `observation.is_human` column.
3. **Instruction source.** `ZarrDataset` falls back to parsing the episode filename (`XXX+instruction+XXX`);
   `MotionTransDataset` requires `task_index` + `meta/tasks.jsonl` (no filename fallback).
4. **Missing-arm handling.** See "Bimanual" above — `ZarrDataset` tolerates missing `robot1_*` per shard,
   `MotionTransDataset` does not.
5. **`camera0_pose` fallback.** Both now fall back to `cam_proj = identity` when the column/key is absent, so
   this one *is* aligned (fixed as part of this LeRobot-support work — previously `MotionTransDataset` loaded
   `observation.camera0_pose` unconditionally and would crash on datasets without it).
6. **Padding order.** `ZarrDataset` pads the action chunk (repeats the last real timestep) *after* converting to
   the relative pose10d representation; `MotionTransDataset` pads the raw pose *before* converting. These
   produce numerically identical results for `pose_rep in {"relative", "abs"}` (padding is just repeating a row
   through a per-row-independent transform), so this is a style difference, not a behavioral one.
