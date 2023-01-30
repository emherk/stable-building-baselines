# %%
import os
import random
from datetime import timedelta
from math import ceil

import numpy as np
import pandas as pd
from meteostat import Hourly, Stations


class OutsideTemp:
    backup_outside_temp_list = [4.3, 4.3, 4.4, 4.4, 4.3, 4.3, 4.7, 5.2, 7.0, 5.3, 8.7, 10.0, 10.7,
                                11.3, 13.6, 10.5, 9.4, 9.0, 9.3, 9.5, 9.3, 9.3, 8.3, 8.3]
    file_name = "outside_temps.csv"
    temperature_df = pd.read_csv(file_name)

    def __init__(self, time_step_size):
        self.time_step_size = time_step_size
        self.interp_df = self.interpolate(time_step_size, self.temperature_df)
        self.time_list = self.interp_df.index.to_list()

    def sample(self, days=timedelta(days=1)):
        """
        This function takes in an integer 'days' and returns a random sample of the temperature data for that many days.
        The sample starts at midnight of a random day in the available temperature data.
        If the requested number of days is greater than the available temperature data, a ValueError is raised.
        """
        hours = days / timedelta(hours=1)
        remaining_temp = len(self.temperature_df) - ceil(
            hours)  # see if the requested number of days fits into the data
        if remaining_temp < 0:
            raise ValueError(
                f"Attempted to run for longer ({days}) than temperature data is available ({len(self.temperature_df)})", )
        remaining_temp = np.interp(remaining_temp, (0, len(self.temperature_df) - 1), (0, len(self.interp_df) - 1))
        random_index = random.randint(0, remaining_temp)
        sample = self.interp_df.loc[random_index:random_index + hours]
        return sample.index.to_list(), sample["temp"].to_list()

    @classmethod
    def interpolate(cls, time_step_size, df):
        """
        Interpolate the temperature data to match the time step size of the simulation.
        """
        df = df.set_index("time")
        freq = f"{time_step_size.total_seconds() / 60:.0f}T"
        return df.resample(freq).interpolate()

    @classmethod
    def _fetch_outside_temperature(cls):
        """
        The function fetch_outside_temperature fetches hourly temperature data from a weather station in Bavaria,
        filters out rows with missing data, and removes rows that are more than an hour away from their adjacent rows.
        The filtered temperature data is stored in the `outside_temp_n` csv file.
        :return:
        """
        # Find and fetch a station in Bavaria
        station_df = Stations().region("DE", "BY").fetch(1)
        station = station_df.iloc[0]
        start_date = station["hourly_start"]
        end_date = station["hourly_end"]

        # Fetch the hourly values and filter out NaN
        hourly_df = Hourly(station["wmo"], start_date, end_date).fetch()
        hourly_df = hourly_df.filter(items=["temp"])
        hourly_df = hourly_df.loc[pd.notnull(hourly_df["temp"])]

        # filter rows which are more than an hour away from previous row (NaN in that hour)
        hourly_df["time_diff"] = hourly_df.index.to_series().diff().dt.seconds
        hourly_df = hourly_df[hourly_df["time_diff"] <= 3610]
        hourly_df = hourly_df.drop(columns=["time_diff"])

        # Save the fetched data to a csv, with a new name
        i = 0
        while os.path.exists(f"{cls.file_name}_{i}.csv"):
            i += 1

        hourly_df.to_csv(f"{cls.file_name}_{i}.csv", index=True)
