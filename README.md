# AgentAccess

**Authors:** [Alexander Michels](https://github.com/alexandermichels)

This project is working towards understanding accessibility through Agent-Based Modeling (ABM).

<hr />

# Functionalities:

## Cost Functions

Our cost functions take in the following arguments:

* time (t) the time from location i to location j
* utilization (u) the demand/supply ratio at location j

and there are a few parameters available for our cost functions:

* beta ($\beta$) for `gravity` - exponent for time
* tau ($\tau$) for `raam` - normalizes time

some parameters are calculated by on input data including:

* rho ($\rho$) for `ramm` - global inverse demand/supply ratio, normalizes supply/demand ratios

Agent seek to minimize the "cost" (in travel/waiting time). The functions we hare implemented are:

* `gravity` given by $$t^{\beta} * u$$
* `raam` given by $$\frac{u}{\rho} + \frac{t}{\tau}$$
* `time` given by $$t^{\beta}$$

## Multiple Travel Modalities

Agents in the framework can have different travel modalities/travel times. This is achieved by setting `cost_df` to a DataFrame with multiple travel-cost columns and then passing a list of columns into `cost_field`. 

* You can set the proportion of agents with each travel modality using the `cost_prop` to pass in a set of weights (summing to one) for each field (where cost_prop[i] is the weight for cost_field[i]).
* You can have the proportion vary by geospatial unit by passing in `cost_prop_fields` and we will assign agents proportionally to these fields for each spatial unit.

## Parameters

The model has a few parameters you can play with:

* `memory`: the number of time steps agents will "remember" when determining which place they should utilize. If memory == 0, they will "remember" everything. 
* `pop_per_agent`: we will give each spatial unit population / pop_per_agent agents rounding up. 
* `top_k`: agents will choose between the top_k best options, proportionally to their cost. So if k=1, they always choose the minimum cost option. If k=3, they choose between the three best options with probability inversely proportional to their costs. For example, if we have locations [i,j,k] with costs [1,2,2] we first invert the costs to get [1, 1/2, 1/2], then normalize to approxiately obtain [0.5, 0.25, 0.25] which are the probabilities the agent chooses [i,j,k] respectively.

<hr />

# Todo:


* Fix code to allow for incomplete travel dicts (i.e. for each GEOID get the possible destinations and only consider resources at those destinations)
* Update statistics to better handle agents who were unable to make a choice

