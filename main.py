import os
from datetime import timedelta

import matplotlib.pyplot as plt
from simplesimple import Building
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from PlottingCallback import PlottingCallback
from building_environement import BuildingEnv

# COMFORT_VALUES = {"COLD": 15, "COOL": 18, "SLIGHT_COOL": 19, "NEUTRAL": 21, "SLIGHT_WARM": 24, "WARM": 26, "HOT": 28}
#
#
# class Stepper:
#     def __init__(self, building: Building, outside_temperature=20, heating_setpoint=18, cooling_setpoint=26):
#         self.building = building
#         self.outside_temperature = outside_temperature
#         self.heating_setpoint = heating_setpoint
#         self.cooling_setpoint = cooling_setpoint
#
#     def step(self):
#         self.building.step(self.outside_temperature, self.heating_setpoint, self.cooling_setpoint)
#
#
# def plotTemp(temperature, thermal_power):
#     # plot the data
#     plt.plot(temperature)
#
#     # add labels and title
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Temperature (Celsius)')
#     plt.title('Temperature Change over Time')
#
#     # display the plot
#     plt.show()
#
#     plt.plot(thermal_power)
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Power (Watts)')
#     plt.title('Power Change over Time')
#
#     # display the plot
#     plt.show()


if __name__ == '__main__':
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

    env = make_vec_env(lambda: env, monitor_dir=log_dir)
    normalized_env = VecNormalize(env)
    model = A2C("MlpPolicy", env).learn(total_timesteps=1000)

    # env = DummyVecEnv([lambda: env], monitor_dir)
    # conditioned_floor_area = 100
    # building = Building(
    #     heat_mass_capacity=165000 * conditioned_floor_area,
    #     heat_transmission=200,
    #     maximum_cooling_power=-10000,
    #     maximum_heating_power=10000,
    #     initial_building_temperature=16,
    #     time_step_size=timedelta(minutes=1),
    #     conditioned_floor_area=conditioned_floor_area
    # )
    #
    # # sample data
    # # simulate one time step
    # print(building.current_temperature)  # returns 16
    # print(building.thermal_power)  # returns 0
    # stepper = Stepper(building)
    # temperature_list = []
    # power_list = []
    # for i in range(100):
    #     stepper.step()
    #     temperature_list.append(building.current_temperature)
    #     power_list.append(building.thermal_power)
    #     # print(building.current_temperature)  # returns ~16.4
    #     # print(building.thermal_power)  # returns 10000
    # plotTemp(temperature_list, power_list)
