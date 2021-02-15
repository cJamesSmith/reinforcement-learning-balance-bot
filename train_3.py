import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench.monitor import Monitor 
import balance_bot
import os

log_dir = "tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('balancebot-v0')
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_balancebot")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs = env.reset()
# while True:
    # action, _states = model.predict(obs)
    # obs, rewards, dones, info = env.step(action)
    # env.render()