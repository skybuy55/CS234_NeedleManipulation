import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.const import ASSET_DIR_PATH
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch.nn as nn
import torch as th
import cv2

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class NeedleReachV1(PsmEnv):
    """
    Refer to Gym FetchReach
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py
    """
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    def _env_setup(self):
        super(NeedleReachV1, self)._env_setup()
        self.has_object = False

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = True

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(NeedleReachV1, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], 3.41057634])#pos[2] ])#+ 0.035 * self.SCALING])
        # print(goal)
        return goal#.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        delta_pos = (obs['desired_goal'] - obs['achieved_goal']) / 0.01
        if np.linalg.norm(delta_pos) < 1.5:
            delta_pos.fill(0)
        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_pos *= 0.3

        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., 0.])
        return action

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal.clone().detach().cpu().numpy(), desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        d = goal_distance(achieved_goal.clone().detach().cpu().numpy(), desired_goal)
        return -d
    
    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        # TODO: may need to modify
        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos = np.array(pos)
            # rotations
            waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
            # relative position state
            object_rel_pos = object_pos - robot_state[0: 3]
        else:
            # TODO: can have a same-length state representation
            object_pos = waypoint_pos = waypoint_rot = object_rel_pos = np.zeros(0)

        if self.has_object:
            # object/waypoint position
            achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos.copy()
        else:
            # tip position
            achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])

        observation = np.concatenate([ # robot_state
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
        # print(observation.shape)
        image = self.render('rgb_array').copy()
        self.image = image.copy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('test_img.png', image)
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :, :, :]
        # print(image.shape)
        desired_goal_est = self.model_adapt(th.from_numpy(image).cuda()).reshape(-1,)
        
        self.feat_adapt = desired_goal_est
        desired_goal_est_cp = desired_goal_est.clone().detach()
        # Unnormalize
        desired_goal_est_cp[0] += 2.6
        desired_goal_est_cp[2] += 3.4356
        self.observation = observation.copy()
        self.achieved_goal = achieved_goal.copy()
        self.desired_goal = self.goal.copy()
        obs = {
            'observation': th.tensor(observation.copy()).cuda(),
            'achieved_goal': th.tensor(achieved_goal.copy()).cuda(),
            'desired_goal': desired_goal_est_cp.clone(),
        }
        return obs


if __name__ == "__main__":
    env = NeedleReach(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
