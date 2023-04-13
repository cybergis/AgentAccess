#!/usr/bin/env python
# coding: utf-8

# # AgentAccess
# 
# **Authors:** Alexander Michels

# In[1]:


import json
import os


# In[2]:


# COMMENT THIS OUT WHEN RUNNING WITH THE SCRIPT
# os.environ["costFunc"] = "raam"
# os.environ["popPerAgent"] = "1000"
# os.environ["timeStep"] = "10"
# os.environ["myK"] = "1"


# In[3]:


# FETCH THE VARS FROM THE ENVIRONMENT
envVariables = ["costFunc", "popPerAgent", "timeStep", "myK"]
paramDict = {}
for var in envVariables:
    paramDict[var] = os.getenv(var)


# In[4]:


int_vars = ["popPerAgent", "timeStep", "myK"]
for iv in int_vars:
    paramDict[iv] = int(paramDict[iv])


# In[5]:


print(json.dumps(paramDict, sort_keys=True, indent=True))


# In[6]:


import access
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import sys
sys.path.append("../")
from agentaccess.Model import AgentAccess
# pandas gives warnings too often
pd.options.mode.chained_assignment = None


# In[7]:


figsize=(10,10)


# ## Load the Data

# In[8]:


pop = gpd.read_file("../data/svi/SVI2018_US_tract.shp")
pop = pop[pop["STCNTY"] == "17031"]
pop = pop.to_crs("EPSG:4326")
print(len(pop))
pop.head()


# In[9]:


pop["vehicle"] = pop["E_TOTPOP"] - pop["E_NOVEH"]


# In[10]:


for x in pop["vehicle"]:
    assert x >= 0


# In[11]:


# load the hospitals
hos = gpd.read_file("../data/hospitals/ContinentalHospitals.shp")
hos = hos[(hos["STATE"] == "IL") & (hos["COUNTY"] == "COOK")]
hos = hos[hos["BEDS"] >= 0]
print(sum(hos["BEDS"]))
hos = hos.to_crs("EPSG:4326")
print(len(hos))
hos.head()


# In[12]:


hos.plot(figsize=figsize)


# In[13]:


# get car travel times
car = pd.read_csv("../data/travel_times/17031-output-TRACT-CAR.csv")
car["origin"] = car["origin"].astype(str)
car["destination"] = car["destination"].astype(str)
car = car.rename(columns={"minutes": "car"})
print(len(car))
car.head()


# In[14]:


# get the transit/walking times
transit = pd.read_csv("../data/travel_times/17031-output-TRACT-TRANSIT,WALK.csv")
transit["origin"] = transit["origin"].astype(str)
transit["destination"] = transit["destination"].astype(str)
transit = transit.rename(columns={"minutes": "transit"})
print(len(transit))
transit.head()


# In[15]:


# get the walking times
walk = pd.read_csv("../data/travel_times/17031-output-TRACT-WALK.csv")
walk["origin"] = walk["origin"].astype(str)
walk["destination"] = walk["destination"].astype(str)
walk = walk.rename(columns={"minutes": "walk"})
print(len(walk))
walk.head()


# In[16]:


travel_times = pd.merge(car, transit, how="outer",
                       left_on=["origin", "destination"],
                       right_on=["origin", "destination"])
print(len(travel_times))
travel_times.head()


# In[17]:


travel_times = pd.merge(travel_times, walk, how="outer",
                       left_on=["origin", "destination"],
                       right_on=["origin", "destination"])
print(len(travel_times))
travel_times.head()


# In[18]:


print(json.dumps(paramDict, sort_keys=True, indent=True))


def average_columns(df, new_column, columns_to_average):
    df[new_column] = df[columns_to_average].mean(axis=1)
    return df

# In[19]:


# intialize our model
# def __init__(self, demand_df, demand_geoid, demand_field, supply_df, supply_geoid, supply_field,
#              cost_df, cost_origin, cost_dest, cost_field, time_steps, memory=0, pop_per_agent=1000, top_k=0):
for i in range(10):
    random.seed(i)
    model = AgentAccess(
        pop,
        "FIPS",
        "E_TOTPOP",
        hos,
        "who cares",
        "BEDS",
        travel_times,
        "origin",
        "destination",
        ["car"],
    #    ["car", "transit"],
        paramDict["timeStep"],
        cost_function=paramDict["costFunc"],
    #     cost_prop=[0.7, 0.2, 0.1],
    #    cost_prop_fields=["vehicle", "E_NOVEH"],
        memory=0,
        pop_per_agent=paramDict["popPerAgent"],
        top_k=paramDict["myK"],
        output_path="../outputs/ParamTest",
        plot=False,
        save_data=False
    )
    model.run_model()
    ts = paramDict["timeStep"]
    portions = [2, 3, 4]
    cols_to_average = ["Cost", "Time", "D/S Ratio"]
    cols_to_save = []
    df = model.spatial_units_df
    for port in portions:
        for col in cols_to_average:
            cols_to_save.append(f"{col}_{port}")
            # average the last 1/port time steps
            df = average_columns(df, f"{col}_{port}", [f"{col} {x}" for x  in range(ts-(ts//port), ts)])
    df = df[cols_to_save]
    df.head()
    df.to_csv(os.path.join(model.output_dir, "spatial_units.csv"))

