import gym
from gym.envs.registration import register
from stable_baselines3 import HerReplayBuffer, SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

register(
    id='NeedleTrack-v1',
    entry_point='surrol.tasks.needle_track_v1:NeedleTrackV1',
    max_episode_steps=50,
)

#here this is needle track.
register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
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
        self.env = gym.make("NeedleTrack-v1")
        self.header = "/mnt/c/users/17599/Desktop/SurRoL-NeedleMove_Visual/logs2/"
        # self.policy_kwargs = dict(
        #         features_extractor_class=CustomCat,
        #         features_extractor_kwargs=dict(features_dim=13),
        #     )
        # self.number = 17500
        # Adaptation model
        self.model_SAC = SAC(
                policy="MultiInputPolicy",
                # policy_kwargs=self.policy_kwargs,
                env=self.env, 
                buffer_size=20000,
                verbose=1, gamma = 0.8)

        # Original policy model
        self.ori_model = SAC(
                policy="MultiInputPolicy",
                env=self.env_ori, 
                buffer_size=20000,
                verbose=1, gamma = 0.8)
        # Load the policy
        self.ori_model = self.ori_model.load("/mnt/c/users/17599/Desktop/SurRoL-NeedleMove/logs_dotonneedle/SAC/20240608state_based10/best_model")
        
        # Load the policy to adptation model (not sure if this is needed, maybe direct using self.ori_model is enough)
        self.params_freeze = self.ori_model.policy.state_dict()
        self.model_SAC.policy.load_state_dict(self.params_freeze, strict=False)
        # self.env.model_adapt = torch.load(f'/mnt/c/users/17599/Desktop/SurRoL-NeedleMove_Visual/logs2/model_adapt_jun11_0{self.number}.pt')
        # self.env.model_adapt.load(torch.load('/mnt/c/users/17599/Desktop/SurRoL-NeedleMove_Visual/logs1/model_adapt_jun11_010000.pt'), strict = False)
        # Optimizer
        self.optim = torch.optim.Adam(self.env.model_adapt.parameters(), lr=1e-5)

    def train(self):
        obs = self.env.reset()
        # i= self.number - 1 #should be 0
        i = 0
        k = 0
        success_best = []
        epoch_train_loss = []
        train_history = []
        episode_reward = []
        success_history = []
        reward_history = []
        rewards = []
        best_reward_1=0
        best_reward_2=0
        best_reward_3=0
        best_reward_4=0
        best_reward_5=0
        while i < 80000:
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

            loss1 = 10*((obs_adapt - obs_gt_desired_goal.detach()) ** 2).sum()
            loss2 = 0.4*((action - action_ori.detach()) ** 2).sum()

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
                # print(obs_gt_desired_goal)
                # print(obs_adapt)
                print(loss1.detach())
                print(loss2.detach())
                # print(episode_reward)
                print('done')
                epoch_train_loss_val = np.array(epoch_train_loss).reshape(-1,).mean()
                train_history.append(epoch_train_loss_val)
                #previous discounted return calculation is inverted.
                discounted_return = 0 #do not discount to match stable baselines3.
                for timestep, reward in enumerate(episode_reward):
                    # discounted_return += reward * (0.99 ** (50 - timestep))
                    discounted_return += reward
                # cur_reward = np.array(episode_reward).reshape(-1,).mean()
                cur_reward = discounted_return
                print(cur_reward)
                # Save the best model within 100000, 200000, 300000 ... timesteps
                if len(rewards) == 0:
                    best_reward_1 = cur_reward
                if (cur_reward > best_reward_1) & (i < 20000):
                    best_reward_1 = cur_reward
                    print('Best', k, best_reward_1)
                    torch.save(self.env.model_adapt, f"{self.header}model_adapt_best_1.pt")
                if (cur_reward > best_reward_2) & ((i >= 20000) & (i < 40000)):
                    best_reward_2 = cur_reward
                    print('Best', k, best_reward_2)
                    torch.save(self.env.model_adapt, f"{self.header}model_adapt_best_2.pt")
                if (cur_reward > best_reward_3) & ((i >= 40000) & (i < 60000)):
                    best_reward_3 = cur_reward
                    print('Best', k, best_reward_3)
                    torch.save(self.env.model_adapt, f"{self.header}model_adapt_best_3.pt")
                if (cur_reward > best_reward_4) & ((i >= 60000) & (i < 80000)):
                    best_reward_4 = cur_reward
                    print('Best', k, best_reward_4)
                    torch.save(self.env.model_adapt, f"{self.header}model_adapt_best_4.pt")
                if (cur_reward > best_reward_5) & (i >= 100000):
                    best_reward_5 = cur_reward
                    print('Best', k, best_reward_5)
                    torch.save(self.env.model_adapt, f"{self.header}model_adapt_best_5.pt")
                rewards.append(cur_reward)
                episode_reward=[]
                epoch_train_loss = []
                obs = self.env.reset()
            eval_interval = 2500
            # Evaluate the model every 10000 timesteps
            if (i % eval_interval == 0):
                self.plot_history(rewards, ind = i // eval_interval)
                torch.save(self.env.model_adapt, f"{self.header}model_adapt_jun11_{i:06d}.pt")
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
                    loss1 = 10*((obs_adapt- obs_gt_desired_goal.detach()) ** 2).sum()
                    loss2 = 0.4*((action - action_ori.detach()) ** 2).sum()
                    loss =  loss1+loss2 
                    action = action.cpu().detach().numpy()
                    obs, reward, done, info = self.env.step(action)
                    episode_reward.append(reward)
                    if done:
                        if info["is_success"]:
                            success_epi += 1
                        j+=1
                        print(info)
                        # print(loss.detach())
                        print('done')
                        obs = self.env.reset()
                        discounted_return = 0
                        for timestep, reward in enumerate(episode_reward):
                            discounted_return += reward
                        print(discounted_return)
                        reward_history.append(discounted_return)
                        episode_reward = []
                success_history.append(success_epi / 20)
                self.plot_success(success_history, reward_history=reward_history, ind = i // eval_interval)
                print('success_epi', success_epi)
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
        while j < 20:
            obs_adapt = self.env.feat_adapt
            obs_gt_desired_goal = torch.tensor(self.env.goal.copy()).cuda()
            obs_gt_desired_goal[0] -= 2.6
            obs_gt_desired_goal[2] -= 3.4356

            loss = ((obs_adapt - obs_gt_desired_goal.detach()) ** 2).mean()
            
            action = self.model_SAC.policy.forward(dict(achieved_goal=obs['achieved_goal'].reshape(1, -1),observation=obs['observation'].reshape(1, -1), desired_goal=obs['desired_goal'].reshape(1, -1)), deterministic=True)
            action = action.cpu().detach().numpy()
            action = action.squeeze()

            obs, reward, done, info = self.env.step(action)
            
            if done:
                if self.env.success_epi is True:
                    success_epi += 1
                j+=1
                print(loss.detach())
                print('done')
                obs = self.env.reset()
        print('success_epi', success_epi)
        
    
    def plot_history(self,train_history, ind):
        # save training curves
        plot_path = f"{self.header}train_reward_{ind}.png"
        plt.figure()
        train_values = train_history
        plt.plot(np.linspace(0, len(train_history)-1, len(train_history)), train_values, label='train')
        plt.tight_layout()
        plt.title('Discounted Reward')
        plt.savefig(plot_path)
        plt.clf()
        print(f'Saved plots')
        
    def plot_success(self,success_history, reward_history, ind):
        # save training curves
        plot_path = f"{self.header}train_success_{ind}.png"
        data_path = f"{self.header}train_success_{ind}.npy"
        reward_path = f"{self.header}train_reward_{ind}.npy"
        plt.figure(2)
        plt.plot(np.linspace(0, len(success_history)-1, len(success_history)), success_history, label='train')
        # plt.tight_layout()
        plt.title('Success Rates')
        plt.savefig(plot_path)
        plt.clf()
        print(f'Saved plots')
        np.save(data_path, success_history) 
        np.save(reward_path, reward_history)

train_adapt = AdaptTrain()
train_adapt.train()