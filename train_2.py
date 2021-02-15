import balance_bot
import gym
# import policy, model and emvironment vectoriser from stable baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# define some hyper parameters
params = {"learning_rate": 1e-3,
          "gamma": 0.99,
          "batch_size": 128}

# initialise the environment
# env = make_vec_env('balancebot-v0', n_envs=1, monitor_dir=log_dir)
# equal to
env = gym.make('balancebot-v0')

# create the agent
agent = DQN('MlpPolicy', DummyVecEnv([lambda: env]), **params)

# reset the environment
obs, state, dones = env.reset(), None, [False]

# train the agent for x amount of timesteps
agent.learn(total_timesteps=200000)
agent.save('sb-balance')

# once the environment is trained, run a while loop
while True:
    actions, state = agent.predict(obs, state=state, mask=dones)
    obs, rew, done, info = env.step(actions)
    # print(obs,rew,done,info)
    # env.render()
    if done:
        env.reset()
        break
env.close()
