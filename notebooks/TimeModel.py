#!/usr/bin/env python
# coding: utf-8

# # Timing AgentAccess for various PopPerAgent Settings
# 
# **Authors:** Alexander Michels
# 
# This notebook walks through the a generalized Agent-Based Model for spatial accessibility analysis.

# In[1]:


import access
import csv
import geopandas as gpd
from IPython.display import Image
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
from tqdm import tqdm
import sys
sys.path.append("../")
from agentaccess.Model import AgentAccess
# pandas gives warnings too often
pd.options.mode.chained_assignment = None


# In[2]:


figsize=(10,10)


# ## Data
# 
# The AgentAccess model has the exact same data requirements as traditional place-based spatial accessibility methods:
# 
# * demand: a population of people needing a good/service
# * supply: a good/service distributed across space
# * spatial impedance: the "cost" to overcome the distances between supply and demand
# 
# We will first load our population dataset, the general population within Cook County, IL.

# In[3]:


pop = gpd.read_file("../data/svi/SVI2018_US_tract.shp")
pop = pop[pop["STCNTY"] == "17031"]
pop = pop.to_crs("EPSG:4326")
print(len(pop))
pop.head()


# The dataset we are using has an estimate of the percentage of households without vehicles (`EPNOVEH`) and we will use this data to simulate agents with and without vehicles in our model:

# In[4]:


pop["no_vehicle"] = (pop["E_TOTPOP"] * pop["EP_NOVEH"] / 100.0).astype(int)
pop["vehicle"] = pop["E_TOTPOP"] - pop["no_vehicle"]
print(sum(pop["E_TOTPOP"]))
pop.head()


# Next we will run a quick sanity check to ensure that we have no negative populations and we have the same number of total population as before:

# In[5]:


for i, row in pop.iterrows():
    assert row["vehicle"] >= 0
    assert row["no_vehicle"] >= 0
    assert row["vehicle"] + row["no_vehicle"] == row["E_TOTPOP"]


# In[6]:


# load the hospitals
hos = gpd.read_file("../data/hospitals/ContinentalHospitals.shp")
hos = hos[(hos["STATE"] == "IL") & (hos["COUNTY"] == "COOK")]
hos = hos[hos["BEDS"] >= 0]
print(sum(hos["BEDS"]))
hos = hos.to_crs("EPSG:4326")
print(len(hos))
hos.head()


# In[7]:


hos.plot(figsize=figsize)


# In[8]:


# get car travel times
car = pd.read_csv("../data/travel_times/17031-output-TRACT-CAR.csv")
car["origin"] = car["origin"].astype(str)
car["destination"] = car["destination"].astype(str)
car = car.rename(columns={"minutes": "car"})
print(len(car))
car.head()


# In[9]:


# get the transit/walking times
transit = pd.read_csv("../data/travel_times/17031-output-TRACT-TRANSIT,WALK.csv")
transit["origin"] = transit["origin"].astype(str)
transit["destination"] = transit["destination"].astype(str)
transit = transit.rename(columns={"minutes": "transit"})
print(len(transit))
transit.head()


# In[10]:


# get the walking times
walk = pd.read_csv("../data/travel_times/17031-output-TRACT-WALK.csv")
walk["origin"] = walk["origin"].astype(str)
walk["destination"] = walk["destination"].astype(str)
walk = walk.rename(columns={"minutes": "walk"})
print(len(walk))
walk.head()


# In[11]:


travel_times = pd.merge(car, transit, how="outer",
                       left_on=["origin", "destination"],
                       right_on=["origin", "destination"])
print(len(travel_times))
travel_times.head()


# In[12]:


travel_times = pd.merge(travel_times, walk, how="outer",
                       left_on=["origin", "destination"],
                       right_on=["origin", "destination"])
print(len(travel_times))
travel_times.head()


# ## Model
# 
# The model executes across a number of time steps set by the user.

# In[ ]:


time_steps = 10
NUM_ITERS = 5
PPA = [1000, 100, 10, 1]


# In[ ]:


with open("outputs/Timing.csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["PopPerAgent", "Iter", "Time"])
    for _itercounter in range(NUM_ITERS):
        for ppa in PPA:
            random.seed(_itercounter)
            _start = time.time()
            # intialize our model
            model = AgentAccess(
                pop,
                "FIPS",
                "E_TOTPOP",
                hos,
                "GEOID",
                "BEDS",
                travel_times,
                "origin",
                "destination",
                ["car", "transit"],
                time_steps,
                cost_prop_fields=["vehicle", "no_vehicle"],
                memory=0,
                pop_per_agent=ppa,
                top_k=5,
                output_path="../outputs/",
                processes=12
            )
            model.run_model()
            _elapsed = time.time() - _start
            print(_elapsed)
            writer.writerow([ppa, _itercounter, _elapsed])


# In[ ]:




