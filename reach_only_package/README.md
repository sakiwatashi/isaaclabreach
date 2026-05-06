# UR10 Reach Only Package

這個包只包含 **UR10 Reach**，可 overlay 到任何已安裝的 IsaacLab / Isaac Sim 5.1 環境。

## 內容
- Reach 任務說明：`UR10_Reach_shareable.md`
- Reach 任務 code overlay：`code/`
- 評估腳本：`eval_success.py`, `eval_reach_visuals.py`

## 使用方式
### 1) 套回 IsaacLab
```bash
cd <reach_only_package> && ./apply_overlay.sh <IsaacLab_path>
```

### 2) 訓練
```bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-UR10-v0 --headless
```

### 3) 評估 success rate
```bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_success.py --task Isaac-Reach-UR10-v0 --num_envs 1024 --headless --num_steps 2000
```

### 4) 產生視覺化評估
```bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_reach_visuals.py --task Isaac-Reach-UR10-Play-v0 --load_run <run_name> --checkpoint <ckpt>.pth --video
```

## 已知前提
- 使用者本機已安裝 Isaac Sim 5.1
- `nvidia-smi` 與 NVIDIA driver 必須正常
- 這個 package 不包含完整 IsaacLab，只放 reach 需要覆蓋的檔案

## 主要結果
- `success_rate: 0.9448`
- `avg_final_distance: 0.0170 m`
