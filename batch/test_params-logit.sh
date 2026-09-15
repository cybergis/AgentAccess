#!/bin/bash
# Author: Alexander Michels
# Purpose: Loops through parameter options and submits jobs with those parameters.

# the cost functions to evaluate
# costFuncs=("gravity" "raam")
costFuncs=("raam")
# the range of time steps to try [start] [count-by] [end]
timeSteps=$(seq 10 5 20)
# the top k options
topK=$(seq 1 5)
# the population-to-agent ratio
numAgents=("1000" "100" "10")
#numAgents=("1000" "100" "10")
# numAgents=("1")
# number of times to perform the analysis for each
# seeds=$(seq 1 5)

# counter should be: 90
# 2 cost funcs x 3 ts x 5 topk x 3 agent

counter=0
for cf in ${costFuncs[@]}; do
  export costFunc=$cf
  for time in ${timeSteps[@]}; do
    export timeStep=$time
    for k in ${topK[@]}; do
      export myK=$k
      for nagents in ${numAgents[@]}; do
        export popPerAgent=$nagents
            printf "Running job with:\n * costFunc: ${costFunc}\n * PopPerAgent: ${popPerAgent}\n * ${seedval}\n * timeStep: ${timeStep}\n * top_k: ${myK}\n"
            sbatch batch/runWithParams-Logit.batch
            let counter+=1
            sleep 1
      done
    done
  done
done

echo "Submitted ${counter} jobs."

