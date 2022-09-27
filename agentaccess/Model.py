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
                 cost_df, cost_origin, cost_dest, cost_field, time_steps, cost_function="raam",
                 beta=2, memory=0, pop_per_agent=1000, tau=30, top_k=0, cost_prop=None, cost_prop_fields=None,
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
        self.cost_field = cost_field if isinstance(cost_field, list) else [cost_field]
        self.cost_prop = cost_prop
        self.cost_prop_fields = cost_prop_fields
        self.time_steps = time_steps
        # parameters
        self.beta = beta
        self.cost_function = cost_function
        self.memory = memory
        self.pop_per_agent = pop_per_agent
        self.top_k = top_k
        # RAAM params
        self.tau = tau
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
        # set rho to the inverse PPR (Provider-to-Patient Ratio)
        self.rho = len(self.demand_df) / sum(self.supply_df["Supply"])
        # print(self.rho)

    def aggregate_resources_to_spatial_units(self) -> None:
        """Takes a GeoDataFrame of resources and spatially joins to spatial units, then aggregates the values giving us a lookup of GEOID -> number of resources"""
        if self.spatial_units_df is None:
            self.get_spatial_units()
        self.supply_df = self.supply_df[self.supply_df[self.supply_field] >= 0]
        self.supply_df = gpd.sjoin(self.spatial_units_df, self.supply_df, how="inner", predicate="intersects")
        self.supply_df = self.supply_df[[self.supply_field, "geometry"]]
        self.supply_df = self.supply_df.rename(columns={self.supply_field: "Supply"})
        self.supply_df = self.supply_df.dissolve(by="GEOID", aggfunc="sum")

    def calculate_costs(self, utilization, time):
        """Calculates the costs for each choice"""
        if self.cost_function == "gravity":
            # inverse of classic gravity
            return (time ** self.beta) * utilization
        elif self.cost_function == "raam":
            # print("Rhos")
            # print((supply / (self.rho)))
            # print("Taus")
            # print(time / self.tau)
            return (utilization / (self.rho)) + (time / self.tau)
        elif self.cost_function == "time":
            return time ** self.beta
        else:  # default to gravity
            return (time ** self.beta) * utilization

    def calculate_mean_cost(self, time):
        """Calculate the agent's average cost per spatial unit"""
        cost_field = f"Cost {time}"
        mean_costs = self.demand_df.groupby(["GEOID"])[cost_field].mean()
        self.spatial_units_df = self.spatial_units_df.drop(columns=[cost_field], errors="ignore")
        self.spatial_units_df = self.spatial_units_df.merge(mean_costs.rename(cost_field).to_frame(), how='left', left_index=True, right_index=True)
        self.spatial_units_df[cost_field] = self.spatial_units_df[cost_field].fillna(0)

    def calculate_mean_ratio_spatial_unit(self, time):
        """Calculate the average supply/demand ratio for each spatial unit"""
        choice_field, ratio_field = f"Choice {time}", f"D/S Ratio {time}"
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

    def calculate_demand_supply_ratio(self, time):
        """Calculates the supply/demand ratio for a given time step"""
        choice, demand, ratio = f"Choice {time}", f"Demand {time}", f"D/S Ratio {time}"
        # figure out how many agents used each supply choice
        utilization = self.demand_df.groupby([choice])["ID"].count()
        # drop the columns if they exist
        self.supply_df = self.supply_df.drop(columns=[demand, ratio], errors="ignore")
        self.supply_df = self.supply_df.merge(utilization.rename(demand).to_frame(), how="left", left_index=True, right_index=True)
        self.supply_df[demand] = self.supply_df[demand].fillna(0)
        # add one to avoid division by zero
        self.supply_df[ratio] = self.supply_df[demand] / self.supply_df["Supply"]

    def calculate_step_statistics(self, time):
        """Calculate statistics for this time step."""
        self.calculate_mean_cost(time)
        self.calculate_demand_supply_ratio(time)
        self.mean_agent_ratio(time)
        self.mean_agent_time(time)

    def choose_resources(self, time_step):
        """Chooses the resource for each agent randomly using gravity models as probability"""
        locations = list(set(self.demand_df["GEOID"]))
        choice_col, cost_col = f"Choice {time_step}", f"Cost {time_step}"
        self.demand_df[choice_col] = ""
        # for geoid in tqdm(locations, desc="Agents choosing", position=0, total=len(locations)):
        for geoid in locations:
            agents_here = self.demand_df[self.demand_df["GEOID"] == geoid].index
            # calculate what is reachable from this location
            reachable = dict()
            for travel_method in self.cost_field:
                travel_from_here = self.cost_df.loc[geoid]
                # get the geoids I can reach
                reachable[travel_method] = travel_from_here[travel_from_here[travel_method].notna()].index
            # get travel time for each method
            times = dict()
            for travel_field in self.cost_field:
                # add the small number to avoid division by zero
                times[travel_field] = np.array([self.cost_df.loc[(geoid, index), travel_field] + 0.1 for index in reachable[travel_field]])
            # calculate the utilization (realistically supply/demand)
            utilization_dict = dict()
            for travel_field in self.cost_field:
                utilization_dict[travel_field] = np.array([self.rho for x in range(len(reachable[travel_field]))])  # use global PPR first time step, ratio afterwards
                if time_step > 0:
                    steps_remembered = range(0, time_step)
                    if self.memory > 0:
                        steps_remembered = range(max(0, time_step - self.memory), time_step)
                    reachable_supplies = self.supply_df[self.supply_df.index.isin(reachable[travel_field])]
                    utilization_dict[travel_field] = np.array(reachable_supplies[[f"D/S Ratio {ts}" for ts in steps_remembered]].mean(axis=1))
            for a_index in agents_here:
                # get the travel method for this agent
                my_cost = self.demand_df.loc[a_index, "Cost Field"]
                # get the times and supplies for all places I can reach
                resources_to_consider = reachable[my_cost]
                if len(resources_to_consider) == 0:
                    self.demand_df.at[a_index, choice_col] = None
                    self.demand_df.at[a_index, cost_col] = None
                    continue
                time = times[my_cost]
                utilization = utilization_dict[my_cost]
                # print(geoid, my_cost, len(reachable[my_cost]), len(reachable[my_cost]), len(time), len(utilization))
                costs = self.calculate_costs(utilization, time)
                if self.top_k > 0:
                    # get the minimum k costs
                    sorted_inds = costs.argsort()
                    ind = sorted_inds[:self.top_k]
                    resources_to_consider = resources_to_consider[ind]
                    costs = costs[ind]
                # probabilities are inverse to costs
                resource_probabilities = 1 / costs
                # print(resource_probabilities)
                resource_probabilities = resource_probabilities / sum(resource_probabilities)
                # print(resource_probabilities)
                choice_ind = np.random.choice(range(len(resource_probabilities)), p=resource_probabilities)
                self.demand_df.at[a_index, choice_col] = resources_to_consider[choice_ind]
                self.demand_df.at[a_index, cost_col] = costs[choice_ind]

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
        if self.cost_prop is not None:
            assert math.isclose(sum(self.cost_prop), 1, rel_tol=0.01)  # use that data, but make sure it sums to 1
        elif self.cost_prop is None:
            # assign equal proportions
            proportions = [1 / len(self.cost_field) for i in self.cost_field]
            self.cost_prop = np.array(proportions)
        agent_counter = 0
        for index, row in self.demand_df.iterrows():
            # get this row's ID
            _id = str(row[self.demand_geoid])
            cprop = self.cost_prop
            if self.cost_prop_fields is not None:
                # create proportions that sum to one based on fields
                props = row[self.cost_prop_fields].values
                # print(props)
                if sum(props) == 0:  # use uniform
                    cprop = [1 / len(self.cost_field) for i in self.cost_field]
                else:
                    cprop = [x / sum(props) for x in props]
                # print(cprop)
            assert math.isclose(sum(cprop), 1, rel_tol=0.01)
            # get the pop data for that locale.
            # Add one so that no location has no agents
            pop_num = float(row[self.demand_field]) + 1
            # determine number of agents in this locale
            num_agents = math.ceil(pop_num / self.pop_per_agent)
            for agent in range(num_agents):
                travel_type = np.random.choice(self.cost_field, p=cprop)
                agent_data.append([agent_counter, _id, travel_type])
                agent_counter += 1
        self.demand_df = pd.DataFrame(agent_data, columns=["ID", "GEOID", "Cost Field"])

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

    def mean_agent_ratio(self, time):
        """Calculate the average supply/demand ratio for each spatial unit"""
        choice_field, ratio_field = f"Choice {time}", f"D/S Ratio {time}"
        choices = self.demand_df[["GEOID", choice_field]].merge(self.supply_df[[ratio_field]], left_on=choice_field, right_index=True)
        mean_ratio = choices.groupby("GEOID")[ratio_field].mean()
        self.spatial_units_df = self.spatial_units_df.merge(mean_ratio.rename(ratio_field).to_frame(), how="left", left_index=True, right_index=True)
        self.spatial_units_df[ratio_field] = self.spatial_units_df[ratio_field].fillna(0)

    def mean_agent_time(self, time):
        """Calculate the average supply/demand ratio for each spatial unit"""
        choice_field, time_field = f"Choice {time}", f"Time {time}"
        choices = self.demand_df[["GEOID", choice_field]].merge(self.cost_df[self.cost_field[0]], left_on=["GEOID", choice_field], right_index=True)
        mean_time = choices.groupby("GEOID")[self.cost_field[0]].mean()
        self.spatial_units_df = self.spatial_units_df.merge(mean_time.rename(time_field).to_frame(), how="left", left_index=True, right_index=True)
        self.spatial_units_df[time_field] = self.spatial_units_df[time_field].fillna(0)

    def plot_this_timestep(self, time, save=True, show=False):
        """Plots the results for this single timestep"""
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        map_axes = [fig.add_subplot(gs[0:2, x]) for x in [0, 1]]
        columns = [f"Cost {time}", f"Time {time}"]
        geodataframes = [self.spatial_units_df, self.spatial_units_df]
        for ax, col, data in zip(map_axes, columns, geodataframes):
            ax = data.plot(ax=ax, column=col, legend=True, legend_kwds={'fraction': 0.046, 'pad': 0.04})
            cx.add_basemap(ax, crs=data.crs.to_string())
            ax.set_title(f"{col[:-2]} at Time Step {time}")
            ax.axis('off')
        chart_axes = [fig.add_subplot(gs[2, x]) for x in [0, 1]]
        # plot the ratio over time
        time_steps = time + 1
        for index, row in self.spatial_units_df.iterrows():
            chart_axes[0].plot(range(time_steps), list(row[[f"D/S Ratio {x}" for x in range(time_steps)]]), alpha=0.2)
            chart_axes[0].axhline(self.rho, linestyle="--", color="r")
            if time > 0:
                chart_axes[0].set_xlim(0, time)
        chart_axes[0].set_title("Average Utilization")
        # plot the time over time
        for index, row in self.spatial_units_df.iterrows():
            chart_axes[1].plot(range(time_steps), list(row[[f"Time {x}" for x in range(time_steps)]]), alpha=0.2)
            if time > 0:
                chart_axes[1].set_xlim(0, time)
        chart_axes[1].set_title("Average Time to Resource")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, f"UI-{time}.png"), dpi=100)
        if show:
            plt.show()
        plt.close('all')

    def run_model(self):
        """Runs the model and saves the outputs"""
        for i in tqdm(range(self.time_steps), position=0, desc="Running Model"):
            self.time_step(i)
        self.save_outputs()
        self.save_gif()

    def save_gif(self):
        """Creates and saves a gif from the outputs"""
        images = []
        for plot in range(self.time_steps):
            images.append(imageio.imread(os.path.join(self.output_dir, "UI-{}.png".format(plot))))
            if plot == self.time_steps - 1:  # if it's the last timestep
                for i in range(6):  # add a three second pause on the result
                    images.append(imageio.imread(os.path.join(self.output_dir, "UI-{}.png".format(plot))))
        imageio.mimsave(os.path.join(self.output_dir, 'movie.gif'), images, fps=2)

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

    def time_step(self, time):
        """Runs a full time step of the model"""
        self.choose_resources(time)
        self.calculate_step_statistics(time)
        self.plot_this_timestep(time)
