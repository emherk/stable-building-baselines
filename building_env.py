# %%
from datetime import timedelta
from functools import partial
from random import randint

import gym
import numpy as np
from numpy import float32

from building_model import Building
from outside_temp import OutsideTemp


class BuildingEnv(gym.Env):
    """
    A gym environment for the Building model.
    """

    metadata = {'render.modes': ['human']}
    COMFORT_PENALTY = -10000
    MAX_TEMP = 42
    MIN_TEMP = 2
    MAX_OUTSIDE_TEMP = 36
    MIN_OUTSIDE_TEMP = -16

    def __init__(
            self, heat_mass_capacity,
            heat_transmission,
            maximum_cooling_power,
            maximum_heating_power,
            time_step: timedelta,
            floor_area,
            episode_length=timedelta(days=2),
    ):
        """
        :param heat_mass_capacity: The heat mass capacity of the building.
        :param heat_transmission: The heat transmission rate of the building.
        :param maximum_cooling_power: The maximum cooling power available to the building.
        :param maximum_heating_power: The maximum heating power available to the building.
        :param time_step: The time step size for the simulation.
        :param floor_area: The floor area that is conditioned by heating and cooling.
        :param episode_length: The length of each episode, defaults to two days.
        :param desired_temperature: The desired temperature in the building, defaults to 21 degrees Celsius.
        """
        # Set up the outside temperature model, and get the initial temperature and time.
        self.energy_reward = 0
        self.comfort_reward = 0
        self.temp_model = OutsideTemp(time_step, episode_length=episode_length)
        self.temp_model.new_sample()
        self.random_temp = partial(randint, 12, 33)

        # Set up the building model.
        self.building = Building(heat_mass_capacity=heat_mass_capacity,
                                 heat_transmission=heat_transmission,
                                 maximum_cooling_power=maximum_cooling_power,
                                 maximum_heating_power=maximum_heating_power,
                                 initial_building_temperature=self.random_temp(),
                                 time_step_size=time_step,
                                 conditioned_floor_area=floor_area)

        # Set up basic variables for the environment.
        self.desired_temperature = 21.5
        self.time_step = time_step
        self.episode_length_steps = episode_length / time_step
        self.i = 0
        self.current_rewards = {}

        # Set up the action and observation spaces for the environment.
        # The observation space is a continuous space, with three observations:
        # building temperature, outside temperature, and current time of day in the format "%H%M".
        self.observation_space = gym.spaces.Box(low=-1, high=+1, shape=(3,))
        self.action_space = gym.spaces.Box(low=-1, high=+1, shape=(1,), dtype=float32)

    def render(self, mode='human'):
        if mode == 'human':
            print(
                f'{{temperature: {self.building.current_temperature:.2f}, power: {self.building.thermal_power:.2f}, outside temp: {self.temp_model.get_temperature(self.i):.2f}'
                f', energy reward: {self.energy_reward:.2f}, comfort reward: {self.comfort_reward:.2f} }}')
        else:
            pass

    def step(self, action):
        self.i += 1
        normalized_power = action[0]
        # un-normalize the power
        power = np.interp(normalized_power, (-1, 1),
                          (self.building.maximum_cooling_power, self.building.maximum_heating_power))
        outside_temperature = self.temp_model.get_temperature(self.i)
        time = self.temp_model.get_time(self.i)

        self.building.step(outside_temperature, power)

        obs = self.normalize_obs(self.building.current_temperature, outside_temperature, time)
        reward = self.calculate_reward()
        done = self.i >= self.episode_length_steps
        info = {}

        return obs, reward, done, info

    def normalize_obs(self, building_temperature, outside_temperature, time):
        # normalize the building temperature
        building_temp_normalized = (building_temperature - self.MIN_TEMP) / (self.MAX_TEMP - self.MIN_TEMP)
        # normalize the outside temperature
        outside_temp_normalized = (outside_temperature - self.MIN_OUTSIDE_TEMP) / (
                self.MAX_OUTSIDE_TEMP - self.MIN_OUTSIDE_TEMP)
        # normalize the time
        time_normalized = time / 2400
        return np.array([building_temp_normalized, outside_temp_normalized, time_normalized], dtype=float32)

    def calculate_reward(self):
        """
        Calculates the reward for the current step.
        The reward is made up of the energy use reward and the comfort reward.
        :return: The reward for the current step.
        """
        self.energy_reward = self.get_energy_use_reward()
        self.comfort_reward = self.get_comfort_reward()

        return (self.energy_reward + self.comfort_reward) / 2

    def get_comfort_reward(self):
        normalized_temp = (self.building.current_temperature - self.desired_temperature) / (
                (self.MAX_TEMP - self.MIN_TEMP) / 6)

        # if it's outside the range of the tanh function, give 0 reward
        if normalized_temp > 1 or normalized_temp < -1:
            scaled_temp = 1
        else:
            scaled_temp = np.abs(np.arctanh(normalized_temp))
        # clip reward at 1
        scaled_temp = min(scaled_temp, 1)
        # flat out the top of the reward curve
        reward = 1 - scaled_temp
        rate = 0.9
        reward = 1 / rate * min(rate, reward)

        return reward

    def get_energy_use_reward(self):
        # normalize the building's thermal power
        normalized_power = abs(self.building.thermal_power) / max(abs(self.building.maximum_heating_power),
                                                                  abs(self.building.maximum_cooling_power))

        return 1 - normalized_power

    def reset(self):
        self.i = 0
        self.temp_model.new_sample()
        time = self.temp_model.get_time(0)
        self.building.current_temperature = self.random_temp()
        self.building.thermal_power = 0
        return self.normalize_obs(self.building.current_temperature, self.temp_model.get_temperature(0), time)
