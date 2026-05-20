"""baseline: A Flower Baseline."""

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from .strategy import FedSaSync
from .utils import train_metrics_aggr_fn

from fedsasync.model import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Read from config
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    strategy_name: str = context.run_config["name"]
    semiasync_deg: int = int(context.run_config["semiasync-deg"])
    dataset_name: str = context.run_config["dataset-name"]
    number_slow: int = int(context.run_config["number-slow"])
    execution_number: int = int(context.run_config["exec-number"])

    # Load global model
    if dataset_name == "uoft-cs/cifar10":
        global_model = Net()
    elif dataset_name == "ylecun/mnist":
        global_model = Net(1, 4)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedSaSync strategy
    strategy = FedSaSync(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_available_nodes=2,
        strategy_name=strategy_name,
        semiasync_deg=semiasync_deg,
        number_slow=number_slow,
        dataset_name=dataset_name,
        execution_number=execution_number,
        train_metrics_aggr_fn=train_metrics_aggr_fn,
    )

    # Start strategy, run FedSaSync for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    """# Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")"""
