import gym
import balance_bot

from baselines import deepq


def main():
    env = gym.make("balancebot-v0")
    act = deepq.load_act(r"C:\Users\xwc\Desktop\rl-learn\balance_act.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        # while not done:
        while True:
            # env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()