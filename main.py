# %%
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from building_env import BuildingEnv
from building_plot_callback import BuildingPlotCallback
# %%

floor_area = 100
env = eval_env = BuildingEnv(
    heat_mass_capacity=165000 * floor_area,
    heat_transmission=200,
    maximum_cooling_power=-10000,
    maximum_heating_power=10000,
    time_step=timedelta(minutes=15),
    floor_area=floor_area,
    episode_length=timedelta(days=15),
)

log_dir = "tmp/gym"
env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)

env = VecNormalize(env)
eval_env = VecNormalize(eval_env)
plot_callback = BuildingPlotCallback(log_dir)

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5,
                                                       min_evals=8,
                                                       verbose=1)
eval_callback = EvalCallback(eval_env,
                             eval_freq=1000,
                             best_model_save_path=log_dir + "/best_model",
                             verbose=1)

env.reset()
env.step(np.ndarray([2000]))
# %%
model = A2C("MlpPolicy", env, tensorboard_log=log_dir)
model.learn(total_timesteps=100, callback=[plot_callback], tb_log_name="A2C_0")

# %%

monitor = env.venv.envs[0]
rewards = monitor.get_episode_rewards()
plt.plot(rewards)
plt.show()

# %%
episode_lengths = monitor.get_episode_lengths()
episode_rewards = monitor.get_episode_rewards()
avg_rewards = [i/j for i, j in zip(episode_rewards, episode_lengths)]
plt.plot(avg_rewards)
plt.yscale("log")
plt.show()

# %%
# plot_output(plot_callback.temperatures, plot_callback.thermal_powers)
temperatures = plot_callback.temperatures
thermal_powers = plot_callback.thermal_powers
BuildingPlotCallback.plot_output_2(temperatures, thermal_powers, label1="Temperature", label2="Thermal power")
plt.show()

# %%
model.save("a2c_0")

floor_area = 100
from datetime import timedelta
from building_model import Building
building = Building(
    heat_mass_capacity=165000 * floor_area,
    heat_transmission=200,
    maximum_cooling_power=-10000,
    maximum_heating_power=10000,
    initial_building_temperature = 10,
    time_step_size=timedelta(minutes=15),
    conditioned_floor_area=floor_area
)

# %%
building.step(12, 10000)
print(building.thermal_power, building.current_temperature)

# %%
model = A2C.load("A2C trained 1M .zip", verbose=1)

# %%

print(model.observation_space, model.action_space)
env
