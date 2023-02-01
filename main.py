# %%
from datetime import timedelta

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
    initial_building_temperature=16,
    time_step=timedelta(minutes=15),
    floor_area=floor_area
)

# %%

log_dir = "tmp/gym"
vec_env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)
# monitor_env = Monitor(env, filename="tmp/gym/monitor.csv")
normalized_env = VecNormalize(vec_env)
eval_env = VecNormalize(eval_env)
plot_callback = BuildingPlotCallback(log_dir)

# Stop training if there is no improvement after more than 3 evaluations
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=1, verbose=1)
eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

# %%
model = A2C("MlpPolicy", normalized_env).learn(total_timesteps=10000, callback=[plot_callback])

# %%
# plot_output(plot_callback.temperatures, plot_callback.thermal_powers)
BuildingPlotCallback.plot_output_2(plot_callback.temperatures[9500:], plot_callback.thermal_powers[9500:])

# %%
plt.plot(plot_callback.thermal_powers[9000:])
plt.show()
