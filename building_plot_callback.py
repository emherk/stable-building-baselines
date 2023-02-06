# %%
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class BuildingPlotCallback(BaseCallback):
    def __init__(self):
        super(BuildingPlotCallback, self).__init__()
        self.temperatures = []
        self.thermal_powers = []
        self.energy_rewards = []
        self.comfort_rewards = []

    def _on_step(self):
        # env = self.model.get_env()
        env = self.training_env

        current_energy_reward = env.get_attr('energy_reward')[0]
        current_comfort_reward = env.get_attr('comfort_reward')[0]
        self.energy_rewards.append(current_energy_reward)
        self.comfort_rewards.append(current_comfort_reward)

        building = env.get_attr('building')[0]
        self.temperatures.append(building.current_temperature)
        self.thermal_powers.append(building.thermal_power)

    def _on_training_end(self):
        self.plot_moving_avg(self.temperatures, "Temperatures", ylabel="Temperature (°C)", title="Temperature")
        self.plot_moving_avg(self.thermal_powers, "Power", ylabel="Power (W)", title="Power")

        rewards = [(x + y) / 2 for x, y in zip(self.energy_rewards, self.comfort_rewards)]

        episode_length = int(self.training_env.get_attr('episode_length_steps')[0])

        episode_temperatures = []
        episode_powers = []
        episode_rewards = []

        total = 0
        # calculate average temperature and power for each episode
        while total + episode_length < len(self.temperatures):
            episode_mean_reward = sum(rewards[total:total + episode_length]) / episode_length
            episode_mean_temperature = sum(self.temperatures[total:total + episode_length]) / episode_length
            episode_mean_power = sum(self.thermal_powers[total:total + episode_length]) / episode_length
            episode_temperatures.append(episode_mean_temperature)
            episode_powers.append(episode_mean_power)
            episode_rewards.append(episode_mean_reward)
            total += episode_length

        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.show()

        # plot average temperature and power for each episode
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Avg Temperature', color=color)
        ax1.plot(episode_temperatures, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Avg Power', color=color)
        ax2.plot(episode_powers, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

    @classmethod
    def plot_moving_avg(cls, values_list, label, moving_avg_window=1000, ylabel=None, title=None):
        # get moving average of rewards with pandas
        series = pd.Series(values_list)
        energy_rewards_moving_avg = series.rolling(moving_avg_window).mean()
        plt.plot(energy_rewards_moving_avg, label=label)
        plt.xlabel('Time (steps)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
