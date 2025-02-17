#!/bin/bash
# Author: Alexander Michels
# Purpose: Loops through parameter options and submits jobs with those parameters.

# the cost functions to evaluate
costFuncs=("gravity")
# costFuncs=("raam")
# the range of time steps to try [start] [count-by] [end]
timeSteps=$(seq 10 5 15)

export seedval="1"
export popPerAgent="1000"
export myK="1"

counter=0
for cf in ${costFuncs[@]}; do
  export costFunc=$cf
  for time in ${timeSteps[@]}; do
    export timeStep=$time
    printf "Running job with:\n * costFunc: ${costFunc}\n * PopPerAgent: ${popPerAgent}\n * ${seedval}\n * timeStep: ${timeStep}\n * top_k: ${myK}\n"
    sbatch batch/runWithParams.batch
    let counter+=1
    sleep 1
  done
done

echo "Submitted ${counter} jobs."

