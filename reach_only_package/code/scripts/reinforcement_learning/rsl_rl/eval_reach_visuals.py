# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained UR10 reach policy and export visualizations.

Outputs:
- rollout videos (optional)
- per-episode trajectory plots
- position / orientation error curves
- JSON and CSV summaries

Example:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_reach_visuals.py \
    --task Isaac-Reach-UR10-Play-v0 \
    --load_run 2026-03-24_14-45-46 \
    --checkpoint last_reach_ur10_ep_1000_rew_0.40511096.pth \
    --video
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from packaging import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip



parser = argparse.ArgumentParser(description="Evaluate a trained UR10 reach policy with visual exports.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Play-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to roll out.")
parser.add_argument("--max_steps", type=int, default=1200, help="Safety cap for steps per episode.")
parser.add_argument("--success_dist", type=float, default=0.02, help="Distance threshold for success accounting.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=1200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_episodes", type=int, default=1, help="Number of episodes to record when video is enabled.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to write plots and summaries.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401

from rsl_rl.runners import DistillationRunner, OnPolicyRunner


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    success: bool
    final_distance: float
    min_distance: float
    final_position_error: float
    final_orientation_error: float


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_episode(output_dir: str, ep_id: int, traj: dict):
    steps = np.asarray(traj["step"], dtype=np.int32)
    ee_pos = np.asarray(traj["ee_pos_w"], dtype=np.float32)
    target_pos = np.asarray(traj["target_pos_w"], dtype=np.float32)
    dist = np.asarray(traj["distance"], dtype=np.float32)
    pos_error = np.asarray(traj["position_error"], dtype=np.float32)
    ori_error = np.asarray(traj["orientation_error"], dtype=np.float32)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], label="ee trajectory", linewidth=2)
    ax1.scatter(target_pos[-1, 0], target_pos[-1, 1], target_pos[-1, 2], c="r", s=80, label="target")
    ax1.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2], c="g", s=50, label="start")
    ax1.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2], c="k", s=50, label="end")
    ax1.set_title(f"Episode {ep_id:02d} trajectory")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(steps, dist, label="distance", color="tab:blue")
    ax2.axhline(args_cli.success_dist, color="tab:red", linestyle="--", label="success threshold")
    ax2.set_title("Distance to target")
    ax2.set_xlabel("step")
    ax2.set_ylabel("m")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(steps, pos_error, label="position error", color="tab:orange")
    ax3.plot(steps, ori_error, label="orientation error", color="tab:green")
    ax3.set_title("Pose errors")
    ax3.set_xlabel("step")
    ax3.set_ylabel("error")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"episode_{ep_id:02d}_visuals.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task.split(":")[-1])
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    if args_cli.output_dir is None:
        output_dir = os.path.join(log_dir, "visuals", "reach_eval")
    else:
        output_dir = args_cli.output_dir
    _ensure_dir(output_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(output_dir, "videos"),
            "episode_trigger": lambda episode_id: episode_id < args_cli.video_episodes,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording rollout video.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    command_mgr = env.unwrapped.command_manager
    body_idx = robot.data.body_names.index("ee_link")

    obs = env.get_observations()
    dt = env.unwrapped.step_dt

    summaries: list[EpisodeSummary] = []
    episode_trajs: list[dict] = []
    current = {"step": [], "ee_pos_w": [], "target_pos_w": [], "distance": [], "position_error": [], "orientation_error": []}
    current_steps = 0
    finished_episodes = 0
    total_steps = 0
    success_count = 0

    while finished_episodes < args_cli.num_episodes and simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            command = command_mgr.get_command("ee_pose")
            des_pos_b = command[:, :3]
            des_quat_b = command[:, 3:]
            des_pos_w, des_quat_w = combine_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                des_pos_b,
                des_quat_b,
            )
            curr_pos_w = robot.data.body_pos_w[:, body_idx]
            curr_quat_w = robot.data.body_quat_w[:, body_idx]
            pos_error, rot_error = compute_pose_error(des_pos_w, des_quat_w, curr_pos_w, curr_quat_w)
            pos_error = torch.norm(pos_error, dim=-1)
            ori_error = torch.norm(rot_error, dim=-1)
            dist = torch.norm(curr_pos_w - des_pos_w, dim=-1)

            current["step"].append(current_steps)
            current["ee_pos_w"].append(curr_pos_w[0].detach().cpu().numpy())
            current["target_pos_w"].append(des_pos_w[0].detach().cpu().numpy())
            current["distance"].append(dist[0].item())
            current["position_error"].append(pos_error[0].item())
            current["orientation_error"].append(ori_error[0].item())
            current_steps += 1
            total_steps += 1

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            done_flag = bool(dones[0].item())
            if done_flag or current_steps >= args_cli.max_steps:
                final_distance = float(current["distance"][-1])
                success = final_distance < args_cli.success_dist
                success_count += int(success)
                summaries.append(
                    EpisodeSummary(
                        episode=finished_episodes,
                        steps=current_steps,
                        success=success,
                        final_distance=final_distance,
                        min_distance=float(np.min(current["distance"])) if current["distance"] else float("nan"),
                        final_position_error=float(current["position_error"][-1]),
                        final_orientation_error=float(current["orientation_error"][-1]),
                    )
                )
                episode_trajs.append({k: list(v) for k, v in current.items()})
                _plot_episode(output_dir, finished_episodes, current)
                current = {"step": [], "ee_pos_w": [], "target_pos_w": [], "distance": [], "position_error": [], "orientation_error": []}
                current_steps = 0
                finished_episodes += 1
                if not done_flag and finished_episodes < args_cli.num_episodes:
                    env.reset()
                    obs = env.get_observations()

    summary_rows = [asdict(s) for s in summaries]
    csv_path = os.path.join(output_dir, "episode_summary.csv")
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": resume_path,
                "task": args_cli.task,
                "num_episodes": len(summaries),
                "success_rate": (success_count / len(summaries)) if summaries else 0.0,
                "episodes": summary_rows,
            },
            f,
            indent=2,
        )
    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    if summaries:
        final_distances = np.asarray([s.final_distance for s in summaries], dtype=np.float32)
        position_errors = np.asarray([s.final_position_error for s in summaries], dtype=np.float32)
        orientation_errors = np.asarray([s.final_orientation_error for s in summaries], dtype=np.float32)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].bar(range(len(summaries)), final_distances)
        ax[0].axhline(args_cli.success_dist, color="r", linestyle="--")
        ax[0].set_title("Final distance")
        ax[1].hist(position_errors, bins=min(10, len(summaries)))
        ax[1].set_title("Final position error")
        ax[2].hist(orientation_errors, bins=min(10, len(summaries)))
        ax[2].set_title("Final orientation error")
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "summary_histograms.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("\n==== UR10 Reach Visual Evaluation ====")
    print(f"checkpoint: {resume_path}")
    print(f"episodes:   {len(summaries)}")
    print(f"success:    {success_count}/{len(summaries)}")
    print(f"output_dir: {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
