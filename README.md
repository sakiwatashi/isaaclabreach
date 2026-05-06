# IsaacLab Reach Only Package

This repository contains a minimal, reach-only package for the UR10 reaching task in IsaacLab.

## What is included
- UR10 Reach task configuration
- PPO training config
- Reach evaluation scripts
- A short shareable task note

## Requirements
- Ubuntu 22.04
- Isaac Sim 5.1 installed
- IsaacLab installed
- NVIDIA driver working (`nvidia-smi` must work)

## Quick start

### 1. Clone this repo
```bash
git clone https://github.com/sakiwatashi/isaaclabreach.git
cd isaaclabreach
```
### 2. Apply the reach overlay to your IsaacLab
bash
cd reach_only_package && ./apply_overlay.sh <IsaacLab_path>
### 3. Train UR10 Reach
bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-UR10-v0 --headless
### 4. Evaluate success rate
bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_success.py --task Isaac-Reach-UR10-v0 --num_envs 1024 --headless --num_steps 2000
### 5. Optional visual evaluation
bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_reach_visuals.py --task Isaac-Reach-UR10-Play-v0 --load_run <run_name> --checkpoint <ckpt>.pth --video
## Notes
- `play.py` is for visualization only, not for quantitative success-rate evaluation.
- The key task is `Isaac-Reach-UR10-v0`.
- The reference baseline reaches about `94.48%` success rate with average final distance around `1.7 cm`.

## Files
- `reach_only_package/README.md`
- `reach_only_package/UR10_Reach_shareable.md`
- `reach_only_package/apply_overlay.sh`
- `reach_only_package/code/`

## License
Add your preferred license here if needed
If you already have IsaacLab 5.1 installed, you only need to apply the overlay and run the training command.
