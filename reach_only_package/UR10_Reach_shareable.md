# UR10 Reach 任務整理（可分享版）

## 1. 任務目標
這條線是 **UR10 的標準 reaching baseline**，目標很單純：

- 讓末端執行器 `ee_link` 穩定到達目標點
- 用可量化的方式評估成功率與最終距離
- 建立後續音訊任務、精細操控任務的前置基礎

這不是抓取任務，也不是音訊任務主線，而是 **機械臂入門級對點 baseline**。

---

## 2. 目前成果
目前這條 reach 線的代表性結果是：

- `success_rate: 0.9448`
- `avg_final_distance: 0.0170124730754178`

也就是說，平均最後距離約 **1.7 公分**，整體成功率約 **94.48%**。

這個結果是目前最值得保留的 reach baseline。

---

## 3. 主要改動
這版能到 94% 不是單靠一個參數，而是幾個方向一起調整：

### 3.1 訓練拉長
- `max_iterations`: `1000 → 2000`

訓練時間拉長後，策略才有足夠機會穩定收斂。

### 3.2 每次更新收更多資料
- `num_steps_per_env`: `24 → 64`

增加每輪收集的步數，讓 PPO 更新更穩定。

### 3.3 調整追蹤平滑度
- `end_effector_position_tracking_fine_grained.std`: `0.05 → 0.06`

讓末端位置追蹤更適合 reaching 的收斂節奏。

### 3.4 加入明確的成功獎勵
新增 sparse `reaching_goal` reward：
- `pos_threshold = 0.02`
- `weight = 8.0`
- `body_names = ["ee_link"]`

這個改動很關鍵，因為它把「真的到位」和「只是接近」分開了。

### 3.5 修正 reward 綁定
`reaching_goal` 明確綁到 `ee_link`，避免 reward 沒落在正確的末端執行器上。

### 3.6 評估方式改正
不要用 `play.py` 看成功率，因為它是展示用，不會正確輸出 success metrics。改用 `eval_success.py` 做量化評估。

---

## 4. 評估方式
### 4.1 正確的評估腳本
```bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_success.py --task Isaac-Reach-UR10-v0 --num_envs 1024 --headless --num_steps 2000
```

### 4.2 訓練腳本
```bash
cd <IsaacLab_path> && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-UR10-v0 --headless
```

> 如果只是要重跑訓練，這個指令就是主入口。

---

## 5. 為什麼這個 baseline 可信
這條 reach 線可信的原因是：

- 成功率有實際數字，不是只看 reward
- 有平均最終距離做輔助判斷
- 評估方式是獨立的 success evaluation，不是訓練曲線自嗨
- 目前結果已經穩定到可作為後續任務基礎

---

## 6. 文件與腳本位置
### 主要記錄文件
- `train.md`

### 評估腳本
- `scripts/reinforcement_learning/rsl_rl/eval_success.py`
- `scripts/reinforcement_learning/rsl_rl/eval_reach_visuals.py`

### 任務入口
- `Isaac-Reach-UR10-v0`

---

## 7. 常見問題
### 7.1 為什麼 `play.py` 不適合拿來看成功率？
因為它主要是視覺展示，通常不會輸出完整 success statistics。

### 7.2 如果訓練一跑就報錯？
先檢查 `reaching_goal` 是否有正確綁定 `ee_link`。

### 7.3 如果評估結果顯示 0 或 nan？
通常代表 metrics 沒有正確寫入，或評估方式不對。

---

## 8. 一句話總結
這條 reach 任務已經不是概念驗證，而是已經有可重現結果的 baseline：

**UR10 reach baseline 成功率 94.48%，平均最終距離約 1.7 公分，可作為後續音訊任務與精細操控的起點。**
