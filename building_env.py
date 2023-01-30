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
                 time_step: timedelta,
                 floor_area,
                 episode_length=timedelta(days=2),
                 desired_temperature=21):
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
        self.temp_model = OutsideTemp(time_step, days=episode_length)
        self.prev_temp = self.temp_model.get_outside_temperature(0)
        self.real_time = self.temp_model.get_time_hours(0)

        # Set up the building model.
        self.building = Building(heat_mass_capacity=heat_mass_capacity,
                                 heat_transmission=heat_transmission,
                                 maximum_cooling_power=maximum_cooling_power,
                                 maximum_heating_power=maximum_heating_power,
                                 initial_building_temperature=self.prev_temp,
                                 time_step_size=time_step,
                                 conditioned_floor_area=floor_area)

        # Set up basic variables for the environment.
        self.desired_temperature = desired_temperature
        self.time_step = time_step
        self.episode_length_steps = episode_length / time_step
        self.i = 0

        self.max_energy_use_heating = maximum_heating_power * self.time_step.total_seconds()

        # Set up the action and observation spaces for the environment.
        # This is a continuous action space, with two actions: heating setpoint and cooling setpoint.
        # The observation space is a continuous space, with three observations:
        # building temperature, thermal power, and current time of day in the format "%H%M".
        self.observation_space = gym.spaces.Box(low=np.array([-30, 0, 0]), high=np.array([60, 10000, 2400]), shape=(3,))
        self.action_space = gym.spaces.Box(low=np.array([7, 15]), high=np.array([30, 45]),
                                           shape=(2,))


    def render(self, mode='human'):
        if mode == 'human':
            print(f'Current temperature: {self.building.current_temperature:.2f}')
            print(f'Thermal power: {self.building.thermal_power:.2f}')
        else:
            pass

    def step(self, action):
        self.i += 1
        heating_setpoint, cooling_setpoint = action
        outside_temperature = self.temp_model.get_outside_temperature(self.i)

        self.prev_temp = self.building.current_temperature
        self.building.step(outside_temperature, heating_setpoint, cooling_setpoint)

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
        energy_use = self.time_step.total_seconds() * self.building.thermal_power
        # Scale reward to [0,3]
        return -np.interp(energy_use, (0, self.max_energy_use_heating), (0, 3))

    def reset(self):
        self.i = 0
        self.building.current_temperature = self.temp_model.sample().get_outside_temperature(0)
        return np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)
