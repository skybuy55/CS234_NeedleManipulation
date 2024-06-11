import gym
from gym.envs.registration import register
from stable_baselines3 import HerReplayBuffer, SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed

register(
    id='NeedleReach-v1',
    entry_point='surrol.tasks.needle_reach_v1:NeedleReachV1',
    max_episode_steps=50,
)

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        layers = []
        units = [256,128,3]
        input_dim=3
        for output_size in units:
            layers.append(nn.Linear(input_dim, output_size))
            layers.append(nn.ReLU())
            input_dim = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        pose=self.mlp(x['desired_goal'])
        return_val = torch.cat((pose, x['observation'], x['achieved_goal']))
        return return_val

class CustomCat(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)

    def forward(self, x):
        return torch.cat((x['observation'], x['achieved_goal'], x['desired_goal']))

class AdaptTrain(object):
    def __init__(self):
        self.env_ori = gym.make("NeedleReach-v0")
        self.env = gym.make("NeedleReach-v1")
        # self.policy_kwargs = dict(
        #         features_extractor_class=CustomCat,
        #         features_extractor_kwargs=dict(features_dim=13),
        #     )
       
        # Adaptation model
        self.model_SAC = SAC(
                policy="MultiInputPolicy",
                # policy_kwargs=self.policy_kwargs,
                env=self.env, 
                verbose=1)

        # Original policy model
        self.ori_model = SAC(
                policy="MultiInputPolicy",
                env=self.env_ori, 
                verbose=1)
        # Load the policy
        self.ori_model = self.ori_model.load("best_model.zip")
        
        # Load the policy to adptation model (not sure if this is needed, maybe direct using self.ori_model is enough)
        self.params_freeze = self.ori_model.policy.state_dict()
        self.model_SAC.policy.load_state_dict(self.params_freeze, strict=False)

        # Optimizer
        self.optim = torch.optim.Adam(self.env.model_adapt.parameters(), lr=5e-5)

        self.success_hist = []
        self.images = []

    def train(self):
        obs = self.env.reset()
        i=0
        k = 0
        success_best = []
        epoch_train_loss = []
        train_history = []
        episode_reward = []
        rewards = []
        best_reward_1=0
        best_reward_2=0
        best_reward_3=0
        best_reward_4=0
        best_reward_5=0
        while i < 50000:
            # Adaptation module output
            obs_adapt = self.env.feat_adapt


            obs_gt_desired_goal = torch.tensor(self.env.goal.copy()).cuda()
            # Normalize the desired goal (Optional)
            obs_gt_desired_goal[0] -= 2.6
            obs_gt_desired_goal[2] -= 3.4356
            
            # Original action with ground-truth desired goal
            action_ori = self.model_SAC.policy.forward(dict(achieved_goal=obs['achieved_goal'].reshape(1, -1),observation=obs['observation'].reshape(1, -1), desired_goal=torch.tensor(self.env.goal.copy()).reshape(1, -1).cuda()),deterministic=True)
            action_ori = action_ori.squeeze()
            # Action with image -> adaptation module -> estimated desired goal
            action = self.model_SAC.policy.forward(dict(achieved_goal=obs['achieved_goal'].reshape(1, -1),observation=obs['observation'].reshape(1, -1), desired_goal=obs['desired_goal'].reshape(1, -1)), deterministic=True)
            action = action.squeeze()

            

            loss1 = 10*((obs_adapt - obs_gt_desired_goal.detach()) ** 2).mean()
            loss2 = 0.3*((action - action_ori.detach()) ** 2).mean()

            # loss function
            loss =  loss1 + loss2 # We can also use loss = loss1

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            epoch_train_loss.append(loss.clone().detach().cpu().numpy())
            action = action.cpu().detach().numpy()
            obs, reward, done, info = self.env.step(action)
            episode_reward.append(reward)
            i+=1
            
            if done:
                k+=1
                print(self.env.success_epi)
                print(obs_gt_desired_goal)
                print(obs_adapt)
                print(loss1.detach())
                print(loss2.detach())
                print('done')
                epoch_train_loss_val = np.array(epoch_train_loss).reshape(-1,).mean()
                train_history.append(epoch_train_loss_val)
                cur_reward = np.array(episode_reward).reshape(-1,).mean()

                rewards.append(cur_reward)
                episode_reward=[]
                epoch_train_loss = []
                obs = self.env.reset()
            
            # Evaluate the model every 10000 timesteps
            if (i % 2000 == 0):
                self.plot_history(rewards)
                torch.save(self.env.model_adapt, 'model_adapt/'+f"model_adapt_{i:06d}.pt")
                obs = self.env.reset()
                self.env.model_adapt.eval()
                j = 0
                success_epi = 0
                while j < 20:
                    obs_adapt = self.env.feat_adapt
                    obs_gt_desired_goal = torch.tensor(self.env.goal.copy()).cuda()
                    obs_gt_desired_goal[0] -= 2.6
                    obs_gt_desired_goal[2] -= 3.4356

                    action_ori = self.model_SAC.policy.forward(dict(achieved_goal=obs['achieved_goal'].reshape(1, -1),observation=obs['observation'].reshape(1, -1), desired_goal=torch.tensor(self.env.goal.copy()).reshape(1, -1).cuda()),deterministic=True)
                    action_ori = action_ori.squeeze()
                    action = self.model_SAC.policy.forward(dict(achieved_goal=obs['achieved_goal'].reshape(1, -1),observation=obs['observation'].reshape(1, -1), desired_goal=obs['desired_goal'].reshape(1, -1)), deterministic=True)
                    action = action.squeeze()
                    loss1 = ((obs_adapt - obs_gt_desired_goal.detach()) ** 2).mean()
                    loss2 = ((action - action_ori.detach()) ** 2).mean()
                    loss =  loss1+loss2 
                    action = action.cpu().detach().numpy()
                    obs, reward, done, info = self.env.step(action)
                    if done:
                        if self.env.success_epi is True:
                            success_epi += 1
                        j+=1
                        print(loss.detach())
                        print('done')
                        obs = self.env.reset()
                print('success_epi', success_epi)
                self.success_hist.append(success_epi)
                np.save('model_adapt/success_rate', np.array(self.success_hist))
                np.save('model_adapt/reward', np.array(rewards))
                success_best.append(success_epi)
                self.env.model_adapt.train()

        print('Best1', best_reward_1)
        print('Best2', best_reward_2)
        print('Best3', best_reward_3)
        print('Best4', best_reward_4)
        print('Best5', best_reward_5)
        print(success_best)
        print(max(success_best))
        print(np.argmax(success_best))
        
    def eval(self):
        obs = self.env.reset()

        self.env.model_adapt.eval()
        success_epi = 0
        j=0
        start_frame = True
        while j < 5:
            obs_adapt = self.env.feat_adapt
            obs_gt_desired_goal = torch.tensor(self.env.goal.copy()).cuda()
            obs_gt_desired_goal[0] -= 2.6
            obs_gt_desired_goal[2] -= 3.4356

            loss = ((obs_adapt - obs_gt_desired_goal.detach()) ** 2).mean()
            # if start_frame:
            #     start = obs['desired_goal'].reshape(1, -1).clone()
            
            action = self.model_SAC.policy.forward(dict(achieved_goal=obs['achieved_goal'].reshape(1, -1),observation=obs['observation'].reshape(1, -1), desired_goal=obs['desired_goal'].reshape(1, -1).clone()), deterministic=True)
            action = action.cpu().detach().numpy()
            action = action.squeeze()

            self.images.append(self.env.image)

            obs, reward, done, info = self.env.step(action)
            
            start_frame = False
            
            if done:
                start_frame = True
                if self.env.success_epi is True:
                    success_epi += 1
                j+=1
                print(loss.detach())
                print('done')
                obs = self.env.reset()
        print('success_epi', success_epi)
        return success_epi
        
    
    def plot_history(self,train_history):
        # save training curves
        plot_path = 'model_adapt/'+'train_reward.png'
        plt.figure()
        train_values = train_history
        plt.plot(np.linspace(0, len(train_history)-1, len(train_history)), train_values, label='train')
        plt.tight_layout()
        plt.title('reward')
        plt.savefig(plot_path)
        plt.clf()
        print(f'Saved plots')

if __name__ == '__main__':
    # import imageio
    # random_seeds = [10, 100, 1000]
    # successes = []
    # train_adapt = AdaptTrain()
    # set_random_seed(100, using_cuda=True)
    # for i in range(3):
    #     set_random_seed(random_seeds[i], using_cuda=True)
    #     # train_adapt.train()
    #     success_num = train_adapt.eval()
    #     successes.append(success_num)
    #     # Complex bent 3
    #     np.save('bent_sealed_tapered2_red', np.array(successes))

    train_adapt = AdaptTrain()
    train_adapt.train()