# %%
from datetime import timedelta

import gym
import numpy as np
from numpy import float32
from simplesimple import Building

from outside_temp import OutsideTemp


class BuildingEnv(gym.Env):
    """
    A gym environment for the Building model.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, heat_mass_capacity,
                 heat_transmission,
                 maximum_cooling_power,
                 maximum_heating_power,
                 initial_building_temperature,
                 time_step_size: timedelta,
                 conditioned_floor_area,
                 episode_length=timedelta(days=2),
                 heating_setpoint=19,
                 cooling_setpoint=26,
                 desired_temperature=21):

        self.building = Building(heat_mass_capacity,
                                 heat_transmission,
                                 maximum_cooling_power,
                                 maximum_heating_power,
                                 initial_building_temperature,
                                 time_step_size,
                                 conditioned_floor_area)

        self.desired_temperature = desired_temperature
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint

        self.time_step_size = time_step_size.total_seconds()
        self.i = 0
        self.initial_building_temperature = initial_building_temperature
        self.max_energy_use_heating = maximum_heating_power * self.time_step_size
        self.observation_space = gym.spaces.Box(low=np.array([-30, 0, 0]), high=np.array([60, 10000, 2400]), shape=(3,))
        self.action_space = gym.spaces.Box(low=np.array([7, 15]), high=np.array([30, 45]),
                                           shape=(2,))
        self.episode_length_days = episode_length
        self.episode_length_steps = self.episode_length_days.total_seconds() / self.time_step_size
        self.outside_temp_list = OutsideTemp.sample(days=self.episode_length_days)
        self.real_time, self.outside_temperature = self.get_outside_temperature()

        self.prev_temp = initial_building_temperature

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Current temperature: {self.building.current_temperature:.2f}')
            print(f'Thermal power: {self.building.thermal_power:.2f}')
            print(f'Heating setpoint: {self.heating_setpoint:.2f}')
            print(f'Cooling setpoint: {self.cooling_setpoint:.2f}')
            print(f'Max cooling power: {self.building.__max_cooling_power:.2f}')
            print(f'Max heating power: {self.building.__max_heating_power:.2f}')
        else:
            pass

    def get_time(self):
        pass
    def get_outside_temperature(self):
        """
        Interpolate temperature from the outside temperature list
        :return: the interpolated temperature
        """
        hours = np.interp(self.i, (0, self.episode_length_steps), (0, self.episode_length_days / timedelta(hours=1)))

        def interpolate(array, index):
            """
            Interpolate a between two values of an array
            :param array: a list of values
            :param index: a float between 0 and len(array) - 1
            :return: the interpolated value between floor(index) and ceil(index)
            """
            i = int(index)
            fraction = index - i
            if i + 1 >= len(array):
                return array[i]
            else:
                return array[i] * (1 - fraction) + array[i + 1] * fraction

        return interpolate(self.outside_temp_list, hours)

    def step(self, action):
        self.i += 1
        heating_setpoint, cooling_setpoint = action
        self.outside_temperature = self.get_outside_temperature()

        self.prev_temp = self.building.current_temperature
        self.building.step(self.outside_temperature, heating_setpoint, cooling_setpoint)

        obs = np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)
        reward = self.calculate_reward()
        done = self.i >= self.episode_length_steps
        info = {"Current temperature": self.building.current_temperature, "Thermal power": self.building.thermal_power}

        return obs, reward, done, info

    def calculate_reward(self):
        return 0.5 * self.get_energy_use_reward() + self.get_comfort_reward() + 2 * self.get_temp_change_reward()

    def get_temp_change_reward(self):
        if self.prev_temp is not None:
            return -abs(self.prev_temp - self.building.current_temperature)
        return 0

    def get_comfort_reward(self):
        return - abs(self.desired_temperature - self.building.current_temperature)

    def get_energy_use_reward(self):
        energy_use = self.time_step_size * self.building.thermal_power
        # Scale reward to [0,3]
        return -np.interp(energy_use, (0, self.max_energy_use_heating), (0, 3))

    def reset(self):
        self.i = 0
        self.building.current_temperature = self.initial_building_temperature
        self.outside_temp_list = OutsideTemp.sample(days=self.episode_length_days)
        return np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)
