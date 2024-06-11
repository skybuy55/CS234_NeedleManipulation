import gym
from gym.envs.registration import register
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'
gym.register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=50,
)
# env = gym.make("NeedlePick-v0")
# # env = SubprocVecEnv([lambda: gym.make("NeedlePick-v0") for _ in range(4)])
# eval_env = gym.make("NeedlePick-v0")
gym.register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)
env = gym.make("NeedleReach-v0")
eval_env = gym.make("NeedleReach-v0")

for seed in [10, 100, 1000]:
    eval_callback = EvalCallback(eval_env, n_eval_episodes = 20, best_model_save_path='/mnt/c/users/17599/Desktop/SurRoL/logs/DDPG/20240608Proprioception_Image'+str(seed) , 
                            log_path='/mnt/c/users/17599/Desktop/SurRoL/logs/DDPG/20240608Proprioception_Image'+str(seed), eval_freq=2500, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=8000, save_path='/mnt/c/users/17599/Desktop/SurRoL/logs/DDPG/20240608Proprioception_Image' +str(seed), name_prefix='rl_model',
                                         save_vecnormalize=True)
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    # model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=20000, replay_buffer_class = None, replay_buffer_kwargs = None, learning_starts = 2000, seed = seed, gamma = 0.8)
    model = DDPG("MultiInputPolicy", env, buffer_size=20000, replay_buffer_class=None, learning_starts = 2000, verbose=1, seed = seed, gamma = 0.8)
    print(model.policy)
    model.learn(total_timesteps=80000, callback=callback)
    