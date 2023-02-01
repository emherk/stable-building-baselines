# %%
from gym.wrappers import Monitor
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from building_env import BuildingEnv


class BuildingPlotCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super(BuildingPlotCallback, self).__init__()
        self.log_dir = log_dir
        self.temperatures = []
        self.thermal_powers = []
        self.rewards = {BuildingEnv.RewardTypes.ENERGY: [],
                        BuildingEnv.RewardTypes.COMFORT: [],
                        BuildingEnv.RewardTypes.CHANGE: []}

    def _on_step(self):
        env = self.model.get_env()

        step = env.get_attr('i')[0]
        if step == 0:
            self.on_reset()

        current_rewards = env.get_attr('current_rewards')[0]
        for key, value in current_rewards.items():
            self.rewards[key].append(value)

        building = env.get_attr('building')[0]
        self.temperatures.append(building.current_temperature)
        self.thermal_powers.append(abs(building.thermal_power))

    def on_reset(self):
        self.temperatures = []
        self.thermal_powers = []
        self.rewards = {BuildingEnv.RewardTypes.ENERGY: [],
                        BuildingEnv.RewardTypes.COMFORT: [],
                        BuildingEnv.RewardTypes.CHANGE: []}

    def _on_training_end(self):
        self.plot_output_2(self.temperatures, self.thermal_powers, "Temperature", "Power")
        self.plot_output_2(self.rewards[BuildingEnv.RewardTypes.ENERGY], self.rewards[BuildingEnv.RewardTypes.COMFORT],
                           "Energy Reward", "Comfort Reward")

        monitor: Monitor = self.model.get_env().venv.envs[0]
        episode_lengths = monitor.get_episode_lengths()
        episode_rewards = monitor.get_episode_rewards()
        avg_rewards = [i / j for i, j in zip(episode_rewards, episode_lengths)]
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.show()

        episode_temperatures = []
        episode_powers = []

        total = 0
        for length in episode_lengths:
            episode_mean_temperature = sum(self.temperatures[total:total + length]) / length
            episode_mean_power = sum(self.thermal_powers[total:total + length]) / length
            episode_temperatures.append(episode_mean_temperature)
            episode_powers.append(episode_mean_power)
            total += length

        self.plot_output_2(episode_temperatures, episode_powers, "Avg Temperature", "Avg Power")

    @classmethod
    def plot_output_2(cls, list1, list2, label1, label2):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Time')
        ax1.set_ylabel(label1, color=color)
        ax1.plot(list1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel(label2, color=color)
        ax2.plot(list2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

    @classmethod
    def plot_output(cls, temperature, thermal_power):
        # plot the data
        plt.plot(temperature, label="Temperature")

        # add labels and title
        plt.xlabel('Time (steps)')
        plt.ylabel('Temperature (Celsius)')
        plt.title('Temperature over Time')

        # display the plot
        plt.show()


        plt.plot(thermal_power)
        plt.xlabel('Time (steps)')
        plt.ylabel('Power (Watts)')
        plt.title('Power over Time')
        plt.show()
