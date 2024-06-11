import time
import socket

import gym
from gym import spaces
from gym.utils import seeding

import os
import subprocess
import xml.etree.ElementTree as ET
import pybullet as p
import pybullet_data
import pkgutil
from surrol.utils.pybullet_utils import (
    step,
    render_image,
)
import numpy as np

RENDER_HEIGHT = 480  # train
RENDER_WIDTH = 640
# RENDER_HEIGHT = 1080  # record
# RENDER_WIDTH = 1920


class SurRoLEnv(gym.Env):
    """
    A gym Env wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'img_array']}

    def __init__(self, render_mode: str = None):
        # rendering and connection options
        self._render_mode = render_mode
        # self._render_mode = 'human'
        # render_mode = 'human'
        self.max_episode_length = 50
        self.current_step = 0
        # if render_mode == 'human':
        #     self.cid = p.connect(p.SHARED_MEMORY)
        #     if self.cid < 0:
        if render_mode == 'human':
            self.cid = p.connect(p.GUI)
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.cid = p.connect(p.DIRECT)
            # See PyBullet Quickstart Guide Synthetic Camera Rendering
            
            if socket.gethostname().startswith('pc') or True:
                # os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
                # os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
                # # Get EGL device
                # assert 'CUDA_VISIBLE_DEVICES' in os.environ
                # devices = os.environ.get('CUDA_VISIBLE_DEVICES', ).split(',')
                # device_id = str(devices[-1])
                # out = subprocess.check_output(['nvidia-smi', '--id='+device_id, '-q', '--xml-format'])
                # tree = ET.fromstring(out)
                # gpu = tree.findall('gpu')[0]
                # dev_id = gpu.find('minor_number').text
                # os.environ['EGL_VISIBLE_DEVICES'] = str(dev_id)
                egl = pkgutil.get_loader('eglRenderer')
                assert egl
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                # egl = pkgutil.get_loader('EglRenderer')
                # plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # camera related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0, 0, 0.2),
                                                                distance=1.5,
                                                                yaw=90,
                                                                pitch=-36,
                                                                roll=0,
                                                                upAxisIndex=2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov=45,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=20.0)
        # additional settings
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf", (0, 0, -0.001))
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        self.seed()

        # self.actions = []  # only for demo
        self._env_setup()
        step(0.25)
        self.goal = self._sample_goal()  # tasks are all implicitly goal-based
        self._sample_goal_callback()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(self.action_size,), dtype='float32')
        if isinstance(obs, np.ndarray):
            # gym.Env
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
        elif isinstance(obs, dict):
            # gym.GoalEnv
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                # image = spaces.Box(0, 255, shape=obs['image'].shape, dtype='uint8'),
            ))
        else:
            raise NotImplementedError

        self._duration = 0.2  # important for mini-steps

    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # time0 = time.time()
        self._set_action(action)
        
        # time1 = time.time()
        # Generate a random force vector in the x and y directions
        force_magnitude = 0.01
        force_direction = np.random.rand(2) - 0.5  # Minus to make it between [-0.5, 0.5]
        force_direction = np.append(force_direction, 0)  # Append 0 to add in the z axis component
        force_direction = force_magnitude * force_direction / np.linalg.norm(force_direction)  # Normalize the x, y components
        # force_direction[-1] = abs(force_direction[-1]) #only change the z axis as I want it to be up. 
        # print("applied force is ", force_direction)
        p.applyExternalForce(self.obj_id, -1, force_direction, [0, 0, 0], p.WORLD_FRAME)
        step(self._duration)
        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        self._step_callback()
        obs = self._get_obs()
        done = False
        self.current_step += 1
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'TimeLimit.truncated': self.current_step >= self.max_episode_length
        } if isinstance(obs, dict) else {'achieved_goal': None}
        if (self.current_step >= self.max_episode_length):
            done = True
            self.current_step = 0
        if isinstance(obs, dict):
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            reward = self.compute_reward(obs, self.goal, info)
        #resemble the goal now, otherwise we are always one step behind.
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()
        # truncated = False #not applicable.
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo
        return obs, reward, done, info

    def reset(self):
        # reset scene in the corresponding file
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

        p.loadURDF("plane.urdf", (0, 0, -0.001))
        self._env_setup()
        step(0.25)
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        obs = self._get_obs()
        info = {}
        return obs

    def close(self):
        if self.cid >= 0:
            p.disconnect()
            self.cid = -1

    def render(self, mode='rgb_array'):
        self._render_callback(mode)
        if mode == "human":
            return np.array([])
        # TODO: check the way to render image
        rgb_array, mask = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                       self._view_matrix, self._proj_matrix)
        if mode == 'rgb_array':
            return rgb_array
        else:
            return rgb_array, mask

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def _env_setup(self):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _set_action(self, action):
        """ Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError

    def _sample_goal(self):
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _sample_goal_callback(self):
        """ For goal visualization, etc.
        """
        pass

    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    @property
    def action_size(self):
        raise NotImplementedError

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError

    def test(self, horizon=100):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        steps, done = 0, False
        obs = self.reset()
        while not done and steps <= horizon:
            tic = time.time()
            action = self.get_oracle_action(obs)
            print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
            # print('action:', action)
            obs, reward, done, info = self.step(action)
            if isinstance(obs, dict):
                print(" -> achieved goal: {}".format(np.round(obs['achieved_goal'], 4)))
                print(" -> desired goal: {}".format(np.round(obs['desired_goal'], 4)))
            else:
                print(" -> achieved goal: {}".format(np.round(info['achieved_goal'], 4)))
            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            toc = time.time()
            print(" -> step time: {:.4f}".format(toc - tic))
            time.sleep(0.05)
        print('\n -> Done: {}\n'.format(done > 0))

    def __del__(self):
        self.close()

