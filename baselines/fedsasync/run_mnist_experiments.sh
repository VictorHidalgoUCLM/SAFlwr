#!/bin/bash

FS_VALUES=(0.0 0.25 0.5)
M_VALUES=(5 7 10)
REPS=5

for fs in "${FS_VALUES[@]}"; do
    for i in $(seq 1 "$REPS"); do
        flwr run . --run-config "dataset-name='ylecun/mnist' name=FedAvg fraction-slow=${fs} exec-number=${i}" --stream

        for m in "${M_VALUES[@]}"; do
            flwr run . --run-config "dataset-name='ylecun/mnist' name=FedSaSync fraction-slow=${fs} semiasync-deg=${m} exec-number=${i}" --stream
        done
    done
done