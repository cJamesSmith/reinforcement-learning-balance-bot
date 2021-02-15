import gym
import balance_bot
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('balancebot-v0')

model = DQN.load(r"C:\Users\xwc\Desktop\rl-learn\sb-balance.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)