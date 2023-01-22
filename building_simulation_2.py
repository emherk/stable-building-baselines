# %%
# %matplotlib notebook

# %%
from datetime import timedelta

import gym
import numpy as np
from numpy import float32
from simplesimple import Building


class BuildingEnv(gym.Env):
    """
    A gym environment for the Building model.
    """

    def __init__(self, heat_mass_capacity, heat_transmission, maximum_cooling_power, maximum_heating_power,
                 initial_building_temperature, time_step_size: timedelta, conditioned_floor_area, heating_setpoint=19,
                 cooling_setpoint=26, outside_temperature=20, desired_temperature=21):
        self.building = Building(heat_mass_capacity, heat_transmission, maximum_cooling_power, maximum_heating_power,
                                 initial_building_temperature, time_step_size, conditioned_floor_area)
        self.desired_temperature = desired_temperature
        self.time_step_size = time_step_size.total_seconds()
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        self.max_cooling_power = maximum_cooling_power
        self.max_heating_power = maximum_heating_power
        self.outside_temperature = outside_temperature
        self.max_energy_use_heating = self.max_heating_power * self.time_step_size
        self.i = 0

        self.observation_space = gym.spaces.Box(low=np.array([-30, 0]), high=np.array([60, 10000]), shape=(2,))
        self.action_space = gym.spaces.Box(low=np.array([10, 40, 100, -100]), high=np.array([30, 45, 10000, -10000]),
                                           shape=(4,))

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Current temperature: {self.building.current_temperature:.2f}')
            print(f'Thermal power: {self.building.thermal_power:.2f}')
            print(f'Heating setpoint: {self.heating_setpoint:.2f}')
            print(f'Cooling setpoint: {self.cooling_setpoint:.2f}')
            print(f'Max cooling power: {self.max_cooling_power:.2f}')
            print(f'Max heating power: {self.max_heating_power:.2f}')
        else:
            pass

    metadata = {'render.modes': ['human']}

    def step(self, action):
        self.i += 1
        heating_setpoint, cooling_setpoint, max_heating_power, max_cooling_power = action
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        self.max_heating_power = max_heating_power
        self.max_cooling_power = max_cooling_power
        self.outside_temperature = self.get_outside_temperature()

        self.building.step(self.outside_temperature, self.heating_setpoint, self.cooling_setpoint)

        obs = np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)
        reward = self.calculate_reward()
        done = self.i >= 100
        info = {"Current temperature": self.building.current_temperature, "Thermal power": self.building.thermal_power}
        print(f"Action: {action}\n"
              f"Current temperature: {self.building.current_temperature}\n"
              f"Thermal power {self.building.thermal_power}")
        return obs, reward, done, info

    def calculate_reward(self):
        return self.get_energy_use_reward() + self.get_comfort_reward()

    def get_comfort_reward(self):
        return - abs(self.desired_temperature - self.building.current_temperature)

    def get_energy_use_reward(self):
        energy_use = self.time_step_size * self.building.thermal_power
        # Scale reward to [0,3]
        return -np.interp(energy_use, (0, self.max_energy_use_heating), (0, 3))

    def reset(self):
        self.building.current_temperature = 20
        self.outside_temperature = 20
        self.i = 0
        return np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)

    def get_outside_temperature(self):
        return self.outside_temperature


# %%
from matplotlib import pyplot as plt


def plotTemp(temperature, thermal_power):
    # plot the data
    plt.plot(temperature)

    # add labels and title
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (Celsius)')
    plt.title('Temperature Change over Time')

    # display the plot
    plt.show()

    plt.plot(thermal_power)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (Watts)')
    plt.title('Power Change over Time')

    plt.show()


# %%
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import ts2xy


# class PlottingCallback(BaseCallback):
#     """
#     Callback for plotting the performance in realtime.
#
#     :param verbose: (int)
#     """
#
#     def __init__(self, verbose=1):
#         super(PlottingCallback, self).__init__(verbose)
#         self._plot = None
#
#     def _on_step(self) -> bool:
#         # get the monitor's data
#         x, y = ts2xy(load_results("tmp/gym"), 'timesteps')
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
#             self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
#                                      self.locals["total_timesteps"] * 1.02])
#             self._plot[-2].autoscale_view(True, True, True)
#             self._plot[-1].canvas.draw()
#

# %%
class BuildingPlotCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super(BuildingPlotCallback, self).__init__()
        self.log_dir = log_dir
        self.temperatures = []
        self.thermal_powers = []

    def _on_step(self) -> bool:
        pass

    def _on_training_end(self):
        print(self.locals)
        x, y = ts2xy(load_results("tmp/gym"), 'timesteps')

        # self.locals['log_info']['ep_info']['temperatures'])
        # thermal_powers = self.locals['log_info']['ep_info']['thermal_powers']

        # plt.figure()
        # plt.plot(temperatures, label='Temperature')
        # plt.plot(thermal_powers, label='Power')
        plt.plot(x, y)
        plt.xlabel('Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(f'{self.log_dir}/building_plot.png')
        plt.close()


# %%
import os
from datetime import timedelta
from stable_baselines3 import A2C

from building_environement import BuildingEnv


def main():
    conditioned_floor_area = 100
    env = BuildingEnv(
        heat_mass_capacity=165000 * conditioned_floor_area,
        heat_transmission=200,
        maximum_cooling_power=-10000,
        maximum_heating_power=10000,
        initial_building_temperature=16,
        time_step_size=timedelta(minutes=15),
        conditioned_floor_area=conditioned_floor_area
    )
    log_dir = "tmp/gym"
    os.makedirs(log_dir, exist_ok=True)

    # vec_env = make_vec_env(lambda: env, n_envs= 1, monitor_dir=log_dir)
    monitor_env = Monitor(env, filename="tmp/gym/monitor.csv")
    # normalized_env = VecNormalize(env)
    # callback = PlottingCallback()
    model = A2C("MlpPolicy", monitor_env).learn(total_timesteps=1000)
    # rewards = monitor_env.env_method("get_episode_rewards")
    print(monitor_env.rewards)
    print(monitor_env)


if __name__ == "__main__":
    main()

# %%
