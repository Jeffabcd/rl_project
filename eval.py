import imageio
from envs import *
from stable_baselines3 import PPO
import cv2
from tqdm import tqdm, trange
def make_gif(agent, file_path):
        env = create_vec_env()
        env.venv.envs[0].game.set_seed(0)
                
        images = []
        frame_proccessor = lambda frame: cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        avg_total_reward=0
        num = 20
        for i in trange(num):
            obs = env.reset()

            done = False
            total_reward=0
            while not done:
                action, _ = agent.predict(obs)
                obs, reward, done, _ = env.step(action)
                state = env.venv.envs[0].game.get_state().screen_buffer
                state = np.moveaxis(state,0,-1)
                total_reward+=reward
                if state.any():
                    images.append(state)
            avg_total_reward+=total_reward/num
            #print(f'total reward for episode: {total_reward}')
        print(images[0].shape)
        print(f'average total reward :{avg_total_reward}')
        imageio.mimsave(file_path, images, duration=30)

        env.close()


model = PPO.load('./train/train_take_cover_n/best_model_126000')
make_gif(model,'figure/take_cover3.gif')


