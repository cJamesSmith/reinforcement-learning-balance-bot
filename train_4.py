import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import balance_bot
import gym
from stable_baselines3.common.vec_env import DummyVecEnv



# class PlottingCallback(BaseCallback):
#     """
#     Callback for plotting the performance in realtime.

#     :param verbose: (int)
#     """

#     def __init__(self, verbose=1):
#         super(PlottingCallback, self).__init__(verbose)
#         self._plot = None

#     def _on_step(self) -> bool:
#         # get the monitor's data
#         x, y = ts2xy(load_results(log_dir), 'timesteps')
#         print(x, y)
#         if self._plot is None:  # make the plot
#             plt.ion()
#             fig = plt.figure(figsize=(6, 3))
#             ax = fig.add_subplot(111)
#             line, = ax.plot(x, y)
#             self._plot = (line, ax, fig)
#             plt.show()
#         else:  # update and rescale the plot
#             self._plot[0].set_data(x, y)
#             self._plot[-2].relim()
#             self._plot[-2].set_xlim([20000 * -0.02,
#                                      20000 * 1.02])
#             self._plot[-2].autoscale_view(True, True, True)
#             self._plot[-1].canvas.draw()


# Create log dir
# log_dir = "tmp/gym/"
# os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
# env = make_vec_env('balancebot-v0', n_envs=1, monitor_dir=log_dir)
env = gym.make('balancebot-v0')
# env = Monitor(env, log_dir)
# env = DummyVecEnv([lambda: env])

# plotting_callback = PlottingCallback()

model = DQN('MlpPolicy', DummyVecEnv([lambda: env]), verbose=0, batch_size=128)
model.learn(total_timesteps=20000,
            # callback=plotting_callback,
            # log_interval=1
            )
