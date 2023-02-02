# %%
from datetime import timedelta
from enum import auto, Enum
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

    class RewardTypes(str, Enum):
        ENERGY = auto()
        COMFORT = auto()
        CHANGE = auto()

    def __init__(
            self, heat_mass_capacity,
            heat_transmission,
            maximum_cooling_power,
            maximum_heating_power,
            time_step: timedelta,
            floor_area,
            episode_length=timedelta(days=2),
            desired_temperature=22,
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
        self.temp_model = OutsideTemp(time_step, episode_length=episode_length)
        self.temp_model.new_sample()
        self.prev_temp = self.temp_model.get_temperature(0)
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
        self.desired_temperature = desired_temperature
        self.time_step = time_step
        self.episode_length_steps = episode_length / time_step
        self.i = 0
        self.current_rewards = {}

        # Set up the action and observation spaces for the environment.
        # The observation space is a continuous space, with three observations:
        # building temperature, outside temperature, and current time of day in the format "%H%M".
        self.observation_space = gym.spaces.Box(low=np.array([-30, -50, 0]), high=np.array([60, 60, 2400]), shape=(3,))
        self.action_space = gym.spaces.Box(low=maximum_cooling_power, high=maximum_heating_power, shape=(1,), dtype=float32)

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Current temperature: {self.building.current_temperature:.2f}')
            print(f'Thermal power: {self.building.thermal_power:.2f}')
        else:
            pass

    def step(self, action):
        self.i += 1
        power = action[0]
        outside_temperature = self.temp_model.get_temperature(self.i)
        time = self.temp_model.get_time(self.i)

        self.prev_temp = self.building.current_temperature
        self.building.step(outside_temperature, power)

        obs = np.array([self.building.current_temperature, outside_temperature, time], dtype=float32)
        reward = self.calculate_reward()
        done = self.i >= self.episode_length_steps
        info = {}
        #     "Current temperature": self.building.current_temperature,
        #     "Thermal power": self.building.thermal_power,
        # }

        return obs, reward, done, info

    def calculate_reward(self):
        r_energy_use = self.get_energy_use_reward()
        r_comfort = self.get_comfort_reward()
        r_temp_change = self.get_temp_change_reward()

        self.current_rewards = {
            self.RewardTypes.ENERGY: r_energy_use,
            self.RewardTypes.COMFORT: r_comfort,
            self.RewardTypes.CHANGE: r_temp_change,
        }

        return r_energy_use + r_comfort + r_temp_change

    def get_temp_change_reward(self):
        return 0
        # if self.prev_temp is None:
        #     return 0
        # return -abs(self.prev_temp - self.building.current_temperature)

    def get_comfort_reward(self):
        if self.desired_temperature - 1 <= self.building.current_temperature <= self.desired_temperature + 1:
            return 0
        return self.COMFORT_PENALTY

    def get_energy_use_reward(self):
        energy_use = abs(self.time_step.total_seconds() * self.building.thermal_power)
        return -energy_use

    def reset(self):
        self.i = 0
        self.temp_model.new_sample()
        self.prev_temp = self.temp_model.get_temperature(0)
        time = self.temp_model.get_time(0)
        self.building.current_temperature = self.random_temp()
        self.building.thermal_power = 0
        obs = np.array([self.building.current_temperature,
                        self.building.thermal_power,
                        time], dtype=float32)
        return obs
