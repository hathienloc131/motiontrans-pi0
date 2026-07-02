# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Pi0-VLA training/inference code for **MotionTrans** (human VR data → robot manipulation policies). It's a fork of
[Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi) (package name is still `openpi`),
extended with a co-training pipeline that mixes human and robot demonstration data. The MotionTrans-specific pieces
live alongside the original openpi code (`aloha`/`droid`/`libero` configs and policies are the untouched upstream
examples; `*_motiontrans` files are the additions for this project).

## Environment & common commands

Package/dependency management is via `uv` (see `pyproject.toml`, `uv.lock`). Python 3.11 (`.python-version`).

```bash
# install deps (per original openpi instructions, not duplicated here)
uv sync

# lint / format (matches pre-commit hooks in .pre-commit-config.yaml)
uv run ruff check --fix .
uv run ruff format .
pre-commit install && pre-commit run --all-files

# run tests (pytest discovers tests under src, scripts, packages)
uv run pytest
uv run pytest src/openpi/models/pi0_test.py            # single file
uv run pytest src/openpi/models/pi0_test.py::test_name  # single test
uv run pytest -m manual                                 # tests marked "manual" are excluded by default; run explicitly
```

The three MotionTrans workflow scripts under `scripts_exp/` are the actual entry points used day-to-day (copy and edit
the env vars / paths at the top before running):

```bash
bash scripts_exp/get_normalize_cotrain.sh   # compute norm stats over the co-training dataset (must run once first)
bash scripts_exp/train_cotrain.sh           # train the pi0_droid_motiontrans co-trained policy
bash scripts_exp/eval.sh                    # offline eval (action MSE) against a checkpoint
bash scripts_exp/serve_policy.sh            # launch the websocket policy server for real-robot inference
```

Each of these ultimately calls one of `scripts/compute_norm_stats.py`, `scripts/train.py`, `scripts/eval.py`,
`scripts/serve_policy.py` with `tyro`-parsed CLI args against a named `TrainConfig` from
`src/openpi/training/config.py` (e.g. `pi0_droid_motiontrans`). `--help` on any of these scripts (via tyro) enumerates
all overridable fields, including nested ones like `--model.max_token_len`.

For real-robot inference: change the checkpoint path in `Checkpoint.dir` for `EnvMode.MOTIONTRANS` in
`scripts/serve_policy.py` (or pass `policy:checkpoint --policy.config=... --policy.dir=...` directly), then run
`serve_policy.sh`. The client side lives in the separate [MotionTrans repo](https://github.com/michaelyuancb/motiontrans).

## Architecture

### Config-driven everything (`src/openpi/training/config.py`)

`TrainConfig` (and the MotionTrans subclass `MotionTransTrainConfig`) is the single source of truth for an
experiment: model architecture, weight loader, optimizer/LR schedule, and `DataConfigFactory`. All named configs are
registered in the `_CONFIGS` list at the bottom of that file and looked up by name via `get_config()` /
`tyro.extras.overridable_config_cli`. To add a new experiment, add a new entry to `_CONFIGS` rather than introducing a
new mechanism.

`DataConfigFactory.create()` builds a `DataConfig`, wiring together three transform groups applied in this order:
`repack_transforms` (dataset-specific key renaming) → `data_transforms` (robot/dataset-specific, e.g.
`MotionTransInputs`/`MotionTransOutputs` in `policies/policy_motiontrans.py`) → normalization → `model_transforms`
(tokenization, common to all robots for a given model type, from `ModelTransformFactory`). The same transform stack
is used for both training (LeRobot/Zarr dataset → model) and inference (robot observation → model → robot action),
which is why transform classes define both `inputs` (dataset/robot → model) and `outputs` (model → robot).

### Two dataset backends, picked at runtime by `dataset_path`

`data_loader.create_dataset()` (`src/openpi/training/data_loader.py`) branches on whether `data_config.dataset_path`
is set:
- **Non-empty `dataset_path`** (the co-training path used by `scripts_exp/*.sh`): uses `ZarrDataset`
  (`policies/dataset_zarr.py`), which reads pre-processed `.zarr` shards (see README's Data Preparation section) and
  supports the `|`-separated multi-folder syntax (e.g. `zarr_data_human|zarr_data_robot`) needed for human+robot
  co-training.
- **Empty `dataset_path`, `dataset_class` set on the data config**: uses the LeRobot-backed `MotionTransDataset`
  (`policies/dataset_motiontrans.py`), which wraps a `LeRobotDataset` and does the same relative-pose/history-window
  processing.
- Otherwise falls back to plain upstream `LeRobotDataset` (aloha/droid/libero configs).

Both MotionTrans dataset classes share the same per-item logic: build an image/state history window (downsampled via
`image_down_sample_steps`/`state_down_sample_steps`), slice + pad an action chunk (downsampled via
`action_down_sample_steps`), and convert absolute end-effector poses to a relative representation via
`policies/pose_repr_util.py` / `policies/pose_util.py` (`proprioception_rep`/`action_rep`, typically `"relative"`).
Camera-pose changes between frames are corrected for by re-projecting poses/actions through
`cam_proj = inv(camera_pose[t]) @ camera_pose[t0]` before computing relative poses.

### Human/robot co-training weighting (`alpha`)

Each dataset item carries an `is_human` flag and emits an `alpha` field (`MotionTransDataset.__getitem__` /
`ZarrDataset`) that rebalances the loss contribution between human and robot samples so overall gradient weight stays
at the configured `alpha` (from `MotionTransTrainConfig.alpha`) regardless of the human/robot sample-count ratio.
`alpha` flows through `MotionTransInputs` into `model.Observation.alpha` and is applied as a per-sample loss weight in
`Pi0.compute_loss` (`src/openpi/models/pi0.py`, look for `observation.alpha`).

### Model layer (`src/openpi/models/`)

`model.BaseModelConfig`/`model.Observation`/`model.Actions` define the model-agnostic interfaces. `pi0.py` (flow-matching
action head + PaliGemma VLM backbone, via flax.nnx) and `pi0_fast.py` (FAST-tokenized autoregressive actions) are the
two `ModelType`s; which one a `TrainConfig` uses is purely a matter of `model=pi0.Pi0Config(...)` vs
`model=pi0_fast.Pi0FASTConfig(...)`. `gemma.py`/`gemma_fast.py`/`siglip.py`/`vit.py` are the backbone building blocks;
`lora.py` implements LoRA variants (`gemma_2b_lora`, `gemma_300m_lora`) used by the `*_low_mem_finetune` configs, whose
`freeze_filter` is obtained from `Pi0Config(...).get_freeze_filter()`.

### Training loop (`scripts/train.py`)

Standard jax/flax.nnx train loop: builds a train state (`init_train_state`), jits `train_step`/`infer_step`, shards
via `training/sharding.py` (supports FSDP through `fsdp_devices`), checkpoints via `training/checkpoints.py`
(orbax-based, supports `--resume`/`--overwrite`), and logs to wandb. When `use_val_dataset` is set, validation runs
`infer_step` (full `sample_actions` rollout, not just teacher-forced loss) and reports action MSE via
`compute_mse`, which reapplies the *inverse* data transforms (`_create_output_transform`) so MSE is measured in
real (unnormalized) action units rather than normalized/tokenized space.

Note: `scripts/train.py` ends with a `notify_task_completion()` call to an internal cluster endpoint gated by
`socket.gethostname()` — this is leftover cluster-specific code from the original authors' training infra, not part
of the general training path; it's a no-op (`exit()`) on any other machine.

### Serving (`scripts/serve_policy.py`, `src/openpi/serving/websocket_policy_server.py`)

A `Policy` (config + checkpoint dir, from `policies/policy_config.py`) is wrapped in a `WebsocketPolicyServer` that
applies the full inference-time transform stack per request. `EnvMode.MOTIONTRANS` is the default env for this repo;
its checkpoint path is a placeholder (`"Put YOUR_CKPT_PATH Here"`) that must be filled in before serving, or overridden
via `--policy.dir`.

### Submodules / assets

`third_party/aloha` and `third_party/libero` are git submodules (only needed for those upstream example envs, not for
MotionTrans training/serving). Norm-stat/action-space conventions for the upstream robot configs are documented in
`docs/norm_stats.md`; remote-inference protocol details are in `docs/remote_inference.md`.
