from vizdoom import *
import random
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3.common import env_checker
import cv2
import os 
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import callbacks
from stable_baselines3.common.evaluation import evaluate_policy
# import ppo for training
from stable_baselines3 import PPO
from stable_baselines3.common import policies
from stable_baselines3.common import vec_env
import random

class VizDoomGym(Env):
    def __init__(self,render=False):
        super().__init__()
        self.game=DoomGame()
        self.game.load_config('github/ViZDoom/scenarios/D3_battle.cfg')
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_window_visible(render)
        self.game.set_death_penalty(2)
        self.game.set_labels_buffer_enabled(True)
        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.action_space = Discrete(5)
        self.frame_proccessor = lambda frame: cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        self.item_count = 0
        self.death_count = 0



    def step(self,action):
        actions = np.identity(5)
        if action == 6:
            reward = self.game.make_action(np.zeros(6),4)
        else:
            reward = self.game.make_action(actions[action],4)
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
        else:
            state = np.zeros(self.observation_space.shape)
        #reward += 2*(self.death_count-death_count)
        reward += self.reward(action)
        info = {"item_count":0,"death_count":0}
        done = self.game.is_episode_finished()

        return state, reward, done, False, info

    def reward(self,action):
        reward=0
        item_count=self.item_count
        death_count=self.death_count
        if self.game.get_state():
            item_count = self.game.get_state().game_variables[3]
            death_count = self.game.get_state().game_variables[4]
            '''
            labels_buffer = self.game.get_state().labels_buffer
            wall = np.count_nonzero(labels_buffer==0)
            if wall/(640*480) >=0.9:
                if action == 3 :
                    reward +=0.3
            '''
        reward += item_count-self.item_count
        return reward

        
    def render():
        pass

    def reset(self,seed=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        ammo = self.game.get_state().game_variables[0]
        info = {"info":ammo}
        return self.grayscale(state), info

    def grayscale(self,observation):
        gray = cv2.cvtColor(np.moveaxis(observation,0 ,-1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation = cv2.INTER_CUBIC)
        state = np.reshape(resize,(100,160,1))
        return state

    def close(self):
        self.game.close()
    def _get_frame(self, done: bool = False) :
        return self.frame_processor( self.game.get_state().screen_buffer) if not done else self.empty_frame


'''
env = VizDoomGym(render=True)
state=env.reset()
while True:
    time.sleep(0.02)
    action = int(input())
    env.step(action)
'''
#env.reset()
#env.close()
#env_checker.check_env(env)
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.fq_count=check_freq

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls >= self.fq_count:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            print(f'best_model_{self.n_calls} save!')
            self.fq_count+=self.check_freq

        return True
def create_env(**kwargs):
    return VizDoomGym()

def create_vec_env(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env(**kwargs)] * n_envs))

CHECKPOINT_DIR = './train/train_D3_battle2'
LOG_DIR = './logs/log_D3_battle'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
#evaluation_callbac
env = create_vec_env(n_envs=4)
env = vec_env.VecNormalize(env,training=True, norm_obs=False, norm_reward=True, clip_reward=2000)
model = PPO(policies.ActorCriticCnnPolicy,env,tensorboard_log=LOG_DIR,verbose=1,learning_rate=0.0001,n_steps=2048)
model = model.load('./train/train_D3_battle2/best_model_50000',env)
model.learn(total_timesteps=600000,callback=callback)
'''
model = PPO.load('./train/train_basic/best_model_10000')
env = VizDoomGym(render = True)
mean_reward, _ = evaluate_policy(model,env,n_eval_episodes=100)
print(f'mean reward: {mean_reward}')
for episode in range(100):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action ,_ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print('Total Reard for episode {} is {}'.format(total_reward, episode))
    time.sleep(2)
'''
