import gym
from gym.envs.registration import register
from stable_baselines3 import HerReplayBuffer, SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from stable_baselines3.common.utils import set_random_seed

set_random_seed(100, using_cuda=True)

env = gym.make("NeedleReach-v0")

eval_callback = EvalCallback(env, n_eval_episodes=20, best_model_save_path='./logs/SAC_image_seed100', 
                            log_path='./logs/', eval_freq=2500, deterministic=True)

checkpoint_callback = CheckpointCallback(save_freq=8000, save_path='./logs/SAC_image_seed100', name_prefix='rl_model',
                                         save_replay_buffer=False, save_vecnormalize=False)

callback = CallbackList([eval_callback, checkpoint_callback])

model = SAC("MultiInputPolicy", env,buffer_size=20000, learning_starts=2000, verbose=1, tensorboard_log='./tensorboard/SAC_image_seed100')
print(model.policy)
model.learn(total_timesteps=80000, callback=callback)