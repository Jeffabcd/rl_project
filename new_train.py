import cv2
#import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import typing as t
import vizdoom
from stable_baselines3 import ppo
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import evaluation, policies
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

#from common import envs, plotting
#import new_shooting_env as envs
#import new_shooting_env as envs
import manager_env as envs
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.LayerNorm([3, 100, 156]),
                                        
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.LayerNorm([32, 24, 38]),
            nn.LeakyReLU(**kwargs),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LayerNorm([64, 11, 18]),
            nn.LeakyReLU(**kwargs),
                                                                                                                
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LayerNorm([64, 9, 16]),
            nn.LeakyReLU(**kwargs),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(9216, features_dim, bias=False),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(**kwargs),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def init_net(m: nn.Module):
    if len(m._modules) > 0:
       for subm in m._modules:
           init_net(m._modules[subm])
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
         nn.init.kaiming_normal_(
         m.weight, 
         a=0.1,         # Same as the leakiness parameter for LeakyReLu.
         mode='fan_in', # Preserves magnitude in the forward pass.
         nonlinearity='leaky_relu')
def init_model(model):
    init_net(model.policy)
def solve_env(env_args, agent_args, n_envs, timesteps, callbacks, eval_freq=None, init_func=None):
    """Helper function to streamline the learning and evaluation process.
    
    Args:
                 env_args:    A dict containing arguments passed to the environment.
         agent_args:  A dict containing arguments passed to the agent.
         n_envs:      The number of parallel training envs to instantiate.
         timesteps:   The number of timesteps for which to train the model.
         callbacks:   A list of callbacks for the training process.
         eval_freq:   The frequency (in steps) at which to evaluate the agent.
         init_func:   A function to be applied on the agent before training.
    """
    # Create environments.
    env = envs.create_vec_env(n_envs, **env_args)

    # Build the agent.
    agent = ppo.PPO(policies.ActorCriticCnnPolicy, env, tensorboard_log='logs/tensorboard', seed=0, **agent_args)
    #agent = ppo.PPO.load('train/store/shooting_7combine',env) 
    # Optional processing on the agent.
    if init_func is not None: 
        init_func(agent)
    # Optional evaluation callback.

    if eval_freq is not None:
        eval_env = envs.create_eval_vec_env(**env_args)

        callbacks.append(EvalCallback(
            eval_env, 
            n_eval_episodes=3, 
            eval_freq=eval_freq, 
            log_path=f'logs/evaluations/{env_args["scenario"]}',
            best_model_save_path=f'logs/models/{env_args["scenario"]}'))

            # Start the training process.
    agent.learn(total_timesteps=timesteps, tb_log_name=env_args['scenario'], callback=callbacks)

    # Cleanup.
    env.close()
    if eval_freq is not None: eval_env.close()
    
    return agent
def frame_processor(frame):
    frame_processor = lambda frame: cv2.resize(frame[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    #print(frame.shape)
    if frame.shape[0] == 3:
        frame = np.moveaxis(frame,0,-1)
    new_frame = frame_processor(frame)
    #print(new_frame.shape)
    return new_frame
    
#frame_processor = lambda frame: cv2.resize(frame[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
env_args = {
    'scenario': 'D3_battle', 
    'frame_skip': 4, 
    'frame_processor': frame_processor
}

agent_args = {
    'n_epochs': 3,
    'n_steps': 2,
    'learning_rate': 1e-4,
    'batch_size': 4,
    'policy_kwargs': {'features_extractor_class': CustomCNN}
}


agent = solve_env(env_args, agent_args, n_envs=2, timesteps=500, callbacks=[], eval_freq=10,init_func = init_model)
