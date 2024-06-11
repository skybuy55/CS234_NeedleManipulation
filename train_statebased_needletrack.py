import gym
from gym.envs.registration import register
from stable_baselines3 import SAC, PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

gym.register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)
env = gym.make("NeedleReach-v0")
eval_env = gym.make("NeedleReach-v0")
    
#seeds previously used are 1, 100, 1000
for seed in [10, 100, 1000]:
    checkpoint_callback = CheckpointCallback(save_freq=8000, save_path='/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/DDPG/20240605state_based' + str(seed), name_prefix='rl_model',
                                         save_replay_buffer=True, save_vecnormalize=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path='/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/DDPG/20240605state_based'+str(seed), 
                            log_path='/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/DDPG/20240605state_based'+str(seed), eval_freq=2500, deterministic=True, n_eval_episodes = 20)
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    model = DDPG("MultiInputPolicy", env, buffer_size=20000, replay_buffer_class = None, verbose=1, seed = seed, learning_starts = 2000, gamma = 0.8)
    print(model.policy)
    model.learn(total_timesteps=80000, callback=callback)

# for seed in [10, 100, 1000]:
#     checkpoint_callback = CheckpointCallback(save_freq=8000, save_path='/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/SAC/20240608state_based' + str(seed), name_prefix='rl_model',
#                                          save_replay_buffer=True, save_vecnormalize=True)
#     eval_callback = EvalCallback(eval_env, best_model_save_path='/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/SAC/20240608state_based'+str(seed), 
#                             log_path='/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/SAC/20240608state_based'+str(seed), eval_freq=2500, deterministic=True, n_eval_episodes = 20)
#     callback = CallbackList([eval_callback, checkpoint_callback])
#     model = SAC("MultiInputPolicy", env, buffer_size=20000, replay_buffer_class = None, verbose=1, seed = seed, learning_starts = 2000, gamma = 0.8)
#     print(model.policy)
#     model.learn(total_timesteps=80000, callback=callback)
    