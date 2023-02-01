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

    def __init__(self, time_step_size, episode_length=timedelta(days=1)):
        self.time_step_size = time_step_size
        self.interp_df = self.interpolate(time_step_size, self.temperature_df)
        self.time_list = self.interp_df.index.to_list()
        self.episode_length = episode_length
        self.start = 0

    def get_temperature(self, i):
        """
        Interpolate temperature from the outside temperature list
        :return: the interpolated temperature
        """
        return self.interp_df.iloc[self.start + i].temp

    def get_time(self, i):
        return int(self.time_list[self.start + i].strftime("%H%M"))

    def new_sample(self):
        episode_length_steps = ceil(self.episode_length / self.time_step_size)

        # see if the requested number of days fits into the data
        remaining_temp = len(self.interp_df) - episode_length_steps
        if remaining_temp < 0:
            raise ValueError(f"Attempted to run for longer ({self.episode_length}) than temperature data is available ({len(self.temperature_df)})")
        self.start = random.randint(0, remaining_temp)

    @classmethod
    def interpolate(cls, time_step_size, df):
        """
        Interpolate the temperature data to match the time step size of the simulation.
        """
        df = df.set_index("time")
        df.index = pd.to_datetime(df.index)
        freq = f"{time_step_size.total_seconds() / 60:.0f}T"
        df = df.resample(freq).interpolate()
        return df

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
