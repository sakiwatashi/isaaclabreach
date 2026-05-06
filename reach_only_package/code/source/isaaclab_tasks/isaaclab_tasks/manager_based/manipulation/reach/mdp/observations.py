# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for reach tasks."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.envs import ManagerBasedRLEnv


def ee_to_command_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "ee_link",
) -> torch.Tensor:
    """Return target position (base frame) minus end-effector position (base frame)."""
    robot = env.scene[robot_cfg.name]
    # end-effector world position
    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, body_idx, :]
    # robot base (root) world pose
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    # ee position in base frame
    ee_pos_base, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, None)
    # command target in base frame
    cmd = env.command_manager.get_command(command_name)
    tgt_pos_base = cmd[:, 0:3]
    # relative vector (target - ee)
    return tgt_pos_base - ee_pos_base
