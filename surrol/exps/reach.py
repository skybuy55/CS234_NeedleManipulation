# from surrol.tasks.needle_regrasp_bimanual import NeedleRegrasp
from surrol.tasks.needle_reach import NeedleReach
import numpy as np
from ddpg import DDPG
import torchvision
from torchvision.models import resnet18
import torch
import torch.nn as nn

# env = NeedleRegrasp(render_mode='rgb_array')
env = NeedleReach(render_mode='rgb_array')

obs = env.reset()
print(obs['img'].shape)

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module,
        predicate,
        func) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# Define different parameters for training the agent
max_episode=100
max_time_steps=5000
ep_r = 0
total_step = 0
score_hist=[]
# for rensering the environmnet
render=True
render_interval=10
# for reproducibility
env.seed(0)
torch.manual_seed(0)
np.random.seed(0)

max_action = float(env.action_space.high[0])

vision_encoder = get_resnet('resnet18')

vision_encoder = replace_bn_with_gn(vision_encoder)

state_dim = 519
action_dim = 5
agent = DDPG(state_dim, action_dim)
for i in range(max_episode):
    total_reward = 0
    step =0
    obs = env.reset()
    img = torch.tensor(obs['img'][:3,:,:]/255.).reshape(1, 3, 128, 128)
    # img_depth = torch.tensor(obs['img'][3,:,:]/255.).reshape(1, 1, 128, 128)
    # img_depth = torch.cat((img_depth, img_depth, img_depth), axis=1)
    # img = torch.cat((img, img_depth), axis=0)
    img = img.to(torch.float32)
    image_features = vision_encoder(img)
    image_features = image_features.reshape(512)
    agent_pos = torch.FloatTensor(obs['observation'])
    state = torch.cat([image_features, agent_pos],dim=-1)
    for  t in range(max_time_steps):
        if t % 100 == 0:
            print(t)
        action = agent.select_action(state)
        # Add Gaussian noise to actions for exploration
        action = (action + np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)
        #action += ou_noise.sample()
        next_state, reward, done, info = env.step(action)
        action = torch.tensor(action)
        # print(action.shape)
        img = torch.tensor(next_state['img'][:3,:,:]/255.).reshape(1, 3, 128, 128)
        # img_depth = torch.tensor(obs['img'][3,:,:]/255.).reshape(1, 1, 128, 128)
        # img_depth = torch.cat((img_depth, img_depth, img_depth), axis=1)
        # img = torch.cat((img, img_depth), axis=0)
        img = img.to(torch.float32)
        image_features = vision_encoder(img)
        image_features = image_features.reshape(512)
        agent_pos = torch.FloatTensor(next_state['observation'])
        next_state = torch.cat([image_features, agent_pos],dim=-1)
        # print('next_state', next_state.shape)
        total_reward += reward
        if render and i >= render_interval : env.render()
        agent.replay_buffer.push((state.detach().numpy(), next_state.detach().numpy(), action, reward, np.float(done)))
        state = next_state
        if done:
            break
        step += 1
        
    score_hist.append(total_reward)
    total_step += step+1
    print("Episode: \t{}  Total Reward: \t{:0.2f}".format( i, total_reward))
    agent.update()
    if i % 10 == 0:
        agent.save()
env.close()