# %%
from datetime import timedelta

from matplotlib import pyplot as plt
from stable_baselines3 import SAC, A2C, DDPG, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from building_env import BuildingEnv
from building_model import Building
from building_plot_callback import BuildingPlotCallback
from outside_temp import OutsideTemp

floor_area = 100
episode_length_real_time = timedelta(days=15)
time_step = timedelta(minutes=15)
episode_length_steps = int(episode_length_real_time / time_step)  # 1440

env = eval_env = BuildingEnv(
    heat_mass_capacity=165000 * floor_area,
    heat_transmission=200,
    maximum_cooling_power=-10000,
    maximum_heating_power=10000,
    time_step=timedelta(minutes=15),
    floor_area=floor_area,
    episode_length=episode_length_real_time,
)

log_dir = "tmp/gym"
plot_callback = BuildingPlotCallback()

# stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5,
#                                                        min_evals=8,
#                                                        verbose=1)

eval_callback = EvalCallback(
    eval_env,
    n_eval_episodes=5,
    eval_freq=episode_length_steps * 5,
    best_model_save_path=log_dir + "/best_model",
    verbose=1
)
# %%
model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log=log_dir + "/PPO",
)
model.learn(total_timesteps=episode_length_steps * 100,
            callback=[eval_callback, plot_callback],
            tb_log_name="PPO",
            reset_num_timesteps=False)


# %%
model.save("DDPG_100_episode_training")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

# %%
temperatures = plot_callback.temperatures
thermal_powers = plot_callback.thermal_powers

# %%
# save plot_callback to a csv file
import pandas as pd

df = pd.DataFrame({'temperatures': temperatures,
                   'thermal_powers': thermal_powers,
                   'comfort_rewards': plot_callback.comfort_rewards,
                   'energy_rewards': plot_callback.energy_rewards})

df.to_csv('SAC_100_episode_training_data.csv', index=False)

 # %%
plot_callback.plot_moving_avg(df['Value'], "PPO Mean Reward", ylabel="Reward", title="PPO Mean Reward")
