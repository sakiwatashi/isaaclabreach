# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # reaching goal reward uses body_names too
        self.rewards.reaching_goal.params["asset_cfg"].body_names = ["ee_link"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR10ReachEnvCfg_MoveSlow(UR10ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # replace command generator with slow moving target
        self.commands.ee_pose = mdp.MovingPoseCommandCfg(
            asset_name="robot",
            body_name="ee_link",
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            ranges=mdp.MovingPoseCommandCfg.Ranges(
                pos_x=(0.48, 0.52),
                pos_y=(-0.03, 0.03),
                pos_z=(0.27, 0.33),
                roll=(0.0, 0.0),
                pitch=(math.pi / 2, math.pi / 2),
                yaw=(-3.14, 3.14),
            ),
            amp_x=0.02,
            amp_y=0.02,
            amp_z=0.02,
            omega=0.15,
        )
        # ensure class_type is bound (safety)
        self.commands.ee_pose.class_type = mdp.MovingPoseCommand
        print("[MoveSlow] ee_pose ranges:", self.commands.ee_pose.ranges)


@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
