import retro #import gym-retro
# import gym
#import policy, model and emvironment vectoriser from stable baselines
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

#define some hyper parameters
params = { "learning_rate": 1e-3,
          "gamma":0.99}

#initialise the environment
env = retro.make('Airstriker-Genesis')

#create the agent
agent = A2C(CnnPolicy,DummyVecEnv([lambda: env]),**params)

#reset the environment
obs, state, dones = env.reset(), None, [False]

#train the agent for x amount of timesteps
agent.learn(total_timesteps = 1000)

#once the environment is trained, run a while loop
while True:
    actions, state = agent.predict(obs, state=state, mask=dones)
    obs,rew,done,info = env.step(actions)
    env.render()
    if done:
        env.reset()
env.close()