#!/bin/bash

FS_VALUES=(0 1 2)
M_VALUES=(6 7 8 9 10)
REPS=10

for fs in "${FS_VALUES[@]}"; do
    for i in $(seq 4 "$REPS"); do
        flwr run . --run-config "name=\"FedAvg\" number-slow=${fs} exec-number=${i}" --stream

        for m in "${M_VALUES[@]}"; do
            flwr run . --run-config "name=\"FedSaSync\" number-slow=${fs} semiasync-deg=${m} exec-number=${i}" --stream
        done
    done
done