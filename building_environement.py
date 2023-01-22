from datetime import timedelta

import gym
import numpy as np
from numpy import float32
from simplesimple import Building


class BuildingEnv(gym.Env):
    """
    A gym environment for the Building model.
    """

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

        self.observation_space = gym.spaces.Box(low=np.array([-30, 0]), high=np.array([60, 10000]), shape=(2,))
        self.action_space = gym.spaces.Box(low=np.array([10, 40, 100, -100]), high=np.array([30, 45, 10000, -10000]),
                                           shape=(4,))

    def step(self, action):
        heating_setpoint, cooling_setpoint, max_heating_power, max_cooling_power = action
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        self.max_heating_power = max_heating_power
        self.max_cooling_power = max_cooling_power
        self.outside_temperature = self.get_outside_temperature()

        self.building.step(self.outside_temperature, self.heating_setpoint, self.cooling_setpoint)

        obs = np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)
        reward = self.calculate_reward()
        done = False
        info = {}
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
        return np.array([self.building.current_temperature, self.building.thermal_power], dtype=float32)

    def get_outside_temperature(self):
        return self.outside_temperature

# Create an instance of the environment
# env = BuildingEnv(heat_mass_capacity=1e6, heat_transmission=1e3, maximum_cooling_power=-1e3, maximum_heating_power=1e3,
#                   initial_building_temperature=20, time_step_size=3600, conditioned_floor_area=100, heating_setpoint=22,
#                   cooling_setpoint=25)

# Wrap the environment in a DummyVecEnv
# env = DummyVecEnv([lambda: env])

# Train the A2C model
# model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)
