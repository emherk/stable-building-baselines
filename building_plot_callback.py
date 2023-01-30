# %%
from stable_baselines3.common.callbacks import BaseCallback


from matplotlib import pyplot as plt
class BuildingPlotCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super(BuildingPlotCallback, self).__init__()
        self.log_dir = log_dir
        self.temperatures = []
        self.thermal_powers = []

    def _on_step(self):
        building = self.model.get_env().get_attr('building')[0]
        self.temperatures.append(building.current_temperature)
        self.thermal_powers.append(building.thermal_power)

    def _on_training_end(self):
        self.plot_output(self.temperatures, self.thermal_powers)


    @classmethod
    def plot_output_2(cls, temperature, thermal_power):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature', color=color)
        ax1.plot(temperature, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Thermal Power', color=color)
        ax2.plot(thermal_power, color=color)
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
