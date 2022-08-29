import contextily as cx
import copy
from datetime import datetime
import geopandas as gpd
import imageio
import json
import math
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# utility function

def is_jsonable(x):
    """Determines if an object is json serializable"""
    try:
        json.dumps(x)
        return True
    except:
        return False


class AgentAccess:
    def __init__(self, demand_df, demand_geoid, demand_field, supply_df, supply_geoid, supply_field,
                 cost_df, cost_origin, cost_dest, cost_field, time_steps, memory=0, pop_per_agent=1000, top_k=0,
                 output_path=None):
        self.demand_df = demand_df
        self.demand_geoid = demand_geoid
        self.demand_field = demand_field
        self.supply_df = supply_df
        self.supply_geoid = supply_geoid
        self.supply_field = supply_field
        self.cost_df = cost_df
        self.cost_origin = cost_origin
        self.cost_dest = cost_dest
        self.cost_field = cost_field
        self.time_steps = time_steps
        # parameters
        self.memory = memory
        self.pop_per_agent = pop_per_agent
        self.top_k = top_k
        # I/O
        self.output_path = output_path
        self.create_output_directory()
        self.save_params()
        # setup data we will work on
        self.spatial_units_df = None
        # prepare data
        self.aggregate_resources_to_spatial_units()
        self.generate_agents_from_pop_data()
        self.get_travel_dict()
        # run the model
        for i in tqdm(range(self.time_steps), position=0, desc="Running Model"):
            self.time_step(i)
        self.save_outputs()
        self.save_gif()

    def aggregate_resources_to_spatial_units(self) -> None:
        """Takes a GeoDataFrame of resources and spatially joins to spatial units, then aggregates the values giving us a lookup of GEOID -> number of resources"""
        if self.spatial_units_df is None:
            self.get_spatial_units()
        self.supply_df = self.supply_df[self.supply_df[self.supply_field] >= 0]
        self.supply_df = gpd.sjoin(self.spatial_units_df, self.supply_df, how="inner", predicate="intersects")
        self.supply_df = self.supply_df[[self.supply_field, "geometry"]]
        self.supply_df = self.supply_df.rename(columns={self.supply_field: "Supply"})
        self.supply_df = self.supply_df.dissolve(by="GEOID", aggfunc="sum")

    def calculate_mean_ratio_spatial_unit(self, time):
        """Calculate the average supply/demand ratio for each spatial unit"""
        choice_field, ratio_field = f"Choice {time}", f"Ratio {time}"
        self.spatial_units_df[ratio_field] = 0
        # for index, row in tqdm(spatial_units.iterrows(), desc="Calc Mean Ratio", position=0, total=len(spatial_units)):
        for index, row in self.spatial_units_df.iterrows():
            # get agents for the location
            location_agents = self.demand_df[self.demand_df["GEOID"] == index]
            agent_choices = list(location_agents[choice_field])
            # calculate average ratio
            ratio_dict = self.supply_df[ratio_field].to_dict()
            avg_ratio = sum([ratio_dict[x] for x in agent_choices]) / len(agent_choices)
            self.spatial_units_df.loc[index, ratio_field] = avg_ratio

    def calculate_supply_demand_ratio(self, time):
        """Calculates the supply/demand ratio for a given time step"""
        choice, demand, ratio = f"Choice {time}", f"Demand {time}", f"Ratio {time}"
        # figure out how many agents used each supply choice
        utilization = self.demand_df.groupby([choice])["ID"].count()
        # drop the columns if they exist
        self.supply_df = self.supply_df.drop(columns=[demand, ratio], errors="ignore")
        self.supply_df = self.supply_df.merge(utilization.rename(demand).to_frame(), how="left", left_on="GEOID", right_index=True)
        self.supply_df[demand] = self.supply_df[demand].fillna(0)
        # add one to avoid division by zero
        self.supply_df[ratio] = self.supply_df["Supply"] / (self.supply_df[demand] + 1)

    def choose_resources_gravity(self, time_step):
        """Chooses the resource for each agent randomly using gravity models as probability"""
        locations = list(set(self.demand_df["GEOID"]))
        choice_field = f"Choice {time_step}"
        self.demand_df[choice_field] = ""
        # for geoid in tqdm(locations, desc="Agents choosing", position=0, total=len(locations)):
        for geoid in locations:
            resource_probabilities = []
            for index, row in self.supply_df.iterrows():
                supply = float(row["Supply"])  # use supply if the first time step, ratio afterwards
                if time_step > 0:
                    # supply = float(row[f"Ratio {time_step-1}"])
                    steps_remembered = range(0, time_step)
                    if self.memory > 0:
                        steps_remembered = range(max(0, time_step - self.memory), time_step)
                    supply = float(np.mean([row[f"Ratio {ts}"] for ts in steps_remembered]))
                # add the small number to avoid division by zero
                time = self.cost_df[geoid, index] + 0.1
                resource_probabilities.append(supply / (time ** 2))
            resource_probabilities = np.asarray(resource_probabilities)
            # print(resource_probabilities)
            resources_to_consider = self.supply_df.index
            if self.top_k >= 0:
                # get the top k options
                ind = np.argpartition(resource_probabilities, - self.top_k)[-self.top_k:]
                resources_to_consider = self.supply_df.index[ind]
                resource_probabilities = resource_probabilities[ind]
            # print(resource_probabilities)
            resource_probabilities = resource_probabilities / sum(resource_probabilities)
            # print(resource_probabilities)
            agents_here = self.demand_df[self.demand_df["GEOID"] == geoid].index
            for a_index in agents_here:
                self.demand_df.at[a_index, choice_field] = np.random.choice(resources_to_consider, p=resource_probabilities)

    def create_output_directory(self):
        """Creates the output directory for the model"""
        self.output_dir = f"AgentAccess-{datetime.now()}"
        if self.output_path is not None:
            self.output_dir = os.path.join(self.output_path, self.output_dir)
        # create the path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_agents_from_pop_data(self):
        """Generates a set of agents based on population data."""
        agent_data = []
        self.demand_df = self.demand_df.drop_duplicates(subset=[self.demand_geoid])
        agent_counter = 0
        for index, row in self.demand_df.iterrows():
            # get this row's ID
            _id = str(row[self.demand_geoid])
            # get the pop data for that locale.
            # Add one so that no location has no agents
            pop_num = float(row[self.demand_field]) + 1
            # determine number of agents in this locale
            num_agents = math.ceil(pop_num / self.pop_per_agent)
            for agent in range(num_agents):
                agent_data.append([agent_counter, _id])
                agent_counter += 1
        self.demand_df = pd.DataFrame(agent_data, columns=["ID", "GEOID"])

    def get_spatial_units(self):
        """
        Gets the spatial units from the population data
        """
        self.spatial_units_df = self.demand_df[[self.demand_geoid, self.demand_field, "geometry"]].copy(deep=True)
        self.spatial_units_df = self.spatial_units_df.rename(columns={self.demand_geoid: "GEOID", self.demand_field: "Demand"})
        self.spatial_units_df = self.spatial_units_df.set_index('GEOID', drop=True)

    def get_travel_dict(self):
        """Gets only the relevant origins and destinations from the cost_df and converts to dict"""
        origins = list(set(self.demand_df["GEOID"]))
        self.cost_df = self.cost_df[self.cost_df[self.cost_origin].isin(origins)]
        destinations = list(set(self.supply_df.index))
        self.cost_df = self.cost_df[self.cost_df[self.cost_dest].isin(destinations)]
        self.cost_df = self.cost_df.set_index([self.cost_origin, self.cost_dest])
        self.cost_df = self.cost_df[self.cost_field].rename("Cost")

    def mean_agent_ratio(self, time):
        """Calculate the average supply/demand ratio for each spatial unit"""
        choice_field, ratio_field = f"Choice {time}", f"Ratio {time}"
        choices = self.demand_df[["GEOID", choice_field]].merge(self.supply_df[[ratio_field]], left_on=choice_field, right_index=True)
        mean_ratio = choices.groupby("GEOID")[ratio_field].mean()
        self.spatial_units_df = self.spatial_units_df.merge(mean_ratio.rename(ratio_field).to_frame(), how="left", left_index=True, right_index=True)
        self.spatial_units_df[ratio_field] = self.spatial_units_df[ratio_field].fillna(0)

    def mean_agent_time(self, time):
        """Calculate the average supply/demand ratio for each spatial unit"""
        choice_field, time_field = f"Choice {time}", f"Time {time}"
        choices = self.demand_df[["GEOID", choice_field]].merge(self.cost_df.to_frame()[["Cost"]], left_on=["GEOID", choice_field], right_index=True)
        mean_time = choices.groupby("GEOID")["Cost"].mean()
        self.spatial_units_df = self.spatial_units_df.merge(mean_time.rename(time_field).to_frame(), how="left", left_index=True, right_index=True)
        self.spatial_units_df[time_field] = self.spatial_units_df[time_field].fillna(0)

    def plot_this_timestep(self, time, save=True, show=True):
        """Plots the results for this single timestep"""
        fig = plt.figure(figsize=(24, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        map_axes = [fig.add_subplot(gs[:, x]) for x in [0, 1]]
        columns = [f"Ratio {time}", f"Time {time}"]
        geodataframes = [self.spatial_units_df, self.spatial_units_df]
        for ax, col, data in zip(map_axes, columns, geodataframes):
            ax = data.plot(ax=ax, column=col, legend=True)
            cx.add_basemap(ax, crs=data.crs.to_string())
            ax.set_title(f"{col[:-2]} at Time Step {time}")
        chart_axes = [fig.add_subplot(gs[x, 2]) for x in [0, 1]]
        # plot the ratio over time
        time_steps = time + 1
        for index, row in self.spatial_units_df.iterrows():
            chart_axes[0].plot(range(time_steps), list(row[[f"Ratio {x}" for x in range(time_steps)]]), alpha=0.2)
        chart_axes[0].set_title("Average Supply-to-Agent Ratio")
        # plot the time over time
        for index, row in self.spatial_units_df.iterrows():
            chart_axes[1].plot(range(time_steps), list(row[[f"Time {x}" for x in range(time_steps)]]), alpha=0.2)
        chart_axes[1].set_title("Average Time to Resource")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, f"UI-{time}.png"))
        if show:
            plt.show()
        plt.close('all')

    def save_gif(self):
        """Creates and saves a gif from the outputs"""
        images = []
        for plot in range(self.time_steps):
            images.append(imageio.imread(os.path.join(self.output_dir, "UI-{}.png".format(plot))))
        imageio.mimsave(os.path.join(self.output_dir, 'movie.gif'), images)

    def save_outputs(self):
        """Saves the output dataframes"""
        # save "regular" dataframes
        to_save = [self.demand_df]
        names = ["demand"]
        for i in range(len(to_save)):
            filename = names[i] + ".csv"
            save_path = os.path.join(self.output_dir, filename)
            to_save[i].to_csv(save_path)
        # save geodataframes
        ext, driver = ".gpkg", "GPKG"
        to_save = [self.supply_df, self.spatial_units_df]
        names = ["supply", "spatial_units"]
        for i in range(len(to_save)):
            filename = names[i] + ext
            save_path = os.path.join(self.output_dir, filename)
            to_save[i].to_file(save_path, driver=driver)

    def save_params(self):
        """Saves the parameters used by the model in the directory"""
        to_save = copy.copy(self.__dict__)
        keys_to_delete = []
        for key, value in to_save.items():
            if not is_jsonable(value):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del to_save[key]
        to_save = json.dumps(to_save, indent=4)
        if self.output_dir is None:
            self.create_output_directory()
        param_path = os.path.join(self.output_dir, "parameters.json")
        with open(param_path, "w") as outfile:
            outfile.write(to_save)

    def time_step(self, time, memory=0, top_k=5):
        """Runs a full time step of the model"""
        self.choose_resources_gravity(time)
        self.calculate_supply_demand_ratio(time)
        self.mean_agent_ratio(time)
        self.mean_agent_time(time)
        self.plot_this_timestep(time)
