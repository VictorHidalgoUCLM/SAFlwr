#!/bin/bash
for fs in 0.0  # Defining fraction-slows
do
    for i in $(seq 1 2) # Define 5 repetitions
    do
        flwr run . --run-config conf/mnist/fedavg_fs${fs}.toml --stream
        flwr run . --run-config conf/mnist/fedsasync_fs${fs}_m5.toml --stream
        flwr run . --run-config conf/mnist/fedsasync_fs${fs}_m8.toml --stream
        flwr run . --run-config conf/mnist/fedsasync_fs${fs}_m10.toml --stream
    done
done