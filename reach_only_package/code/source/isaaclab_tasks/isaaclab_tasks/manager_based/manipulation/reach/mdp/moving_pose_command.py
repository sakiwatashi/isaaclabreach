# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Moving pose command for reach tasks (slow-moving targets)."""

from __future__ import annotations

import math
import torch

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg


@configclass
class MovingPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for moving pose command generator."""

    class_type: type = None  # set in __post_init__

    # movement amplitude (meters)
    amp_x: float = 0.05
    amp_y: float = 0.05
    amp_z: float = 0.03

    # angular speed (rad/s)
    omega: float = 0.2

    def __post_init__(self):
        # ensure class_type is bound
        self.class_type = MovingPoseCommand


class MovingPoseCommand(UniformPoseCommand):
    """Uniform pose command with slow sinusoidal motion in base frame."""

    cfg: MovingPoseCommandCfg

    def __init__(self, cfg: MovingPoseCommandCfg, env):
        super().__init__(cfg, env)
        self._t = torch.zeros(self.num_envs, device=self.device)
        # store base command to oscillate around
        self._base_pose_b = torch.zeros_like(self.pose_command_b)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # reset timer and store base pose
        self._t[env_ids] = 0.0
        self._base_pose_b[env_ids] = self.pose_command_b[env_ids]

    def _update_command(self):
        # advance time
        dt = float(self._env.step_dt)
        self._t += dt

        # sinusoidal offsets in base frame
        dx = self.cfg.amp_x * torch.sin(self.cfg.omega * self._t)
        dy = self.cfg.amp_y * torch.cos(self.cfg.omega * self._t)
        dz = self.cfg.amp_z * torch.sin(self.cfg.omega * self._t * 0.5)

        self.pose_command_b[:, 0] = self._base_pose_b[:, 0] + dx
        self.pose_command_b[:, 1] = self._base_pose_b[:, 1] + dy
        self.pose_command_b[:, 2] = self._base_pose_b[:, 2] + dz

        # clamp to configured ranges
        self.pose_command_b[:, 0] = torch.clamp(self.pose_command_b[:, 0], *self.cfg.ranges.pos_x)
        self.pose_command_b[:, 1] = torch.clamp(self.pose_command_b[:, 1], *self.cfg.ranges.pos_y)
        self.pose_command_b[:, 2] = torch.clamp(self.pose_command_b[:, 2], *self.cfg.ranges.pos_z)


# bind class_type after definition
MovingPoseCommandCfg.class_type = MovingPoseCommand
