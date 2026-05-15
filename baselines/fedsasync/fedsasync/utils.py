"""FedSaSync: A Flower Semi-Asynchronous strategy based on message-based FedAvg aggregation."""
import csv
from logging import INFO
from flwr.common import log
from flwr.serverapp.strategy.result import Result

def save_logs(
        result: Result,
        strategy_name: str,
        semiasync_deg: int,
        fraction_slow: float,
        dataset_name: str,        
    ) -> None:
    """Save the federated result in a csv.

    Utility function to write the federated final result on a csv, defined by
    dataset_name, strategy_name, fraction_slow, and semiasync_deg.

    Parameters
    ----------
    result : Result
        Result object containing metrics collected across rounds.
    strategy_name : str
        Strategy name used to build the output path.
    semiasync_deg : int
        Semi-asynchronous degree (M) of the experiment.
    fraction_slow : float
        Fraction of slow clients in the experiment.
    dataset_name : str
        Dataset name used for the experiment.
    """
    # Map dataset identifiers to short names used in paths
    dataset_map = {
        "uoft-cs/cifar10": "cifar10",
        "ylecun/mnist": "mnist",
    }
    dataset = dataset_map.get(dataset_name, dataset_name)

    # Build output path depending on strategy type
    if strategy_name == "FedSaSync":
        path = (
            f"results/{dataset}/"
            f"{strategy_name}_fs{fraction_slow}_m{semiasync_deg}.csv"
        )
    else:
        path = (
            f"results/{dataset}/"
            f"{strategy_name}_fs{fraction_slow}.csv"
        )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Fixed column order for easier post-processing
        writer.writerow([
            "time",
            "loss",
            "acc",
            "train_loss",
        ])

        # Use rounds available in evaluation metrics
        rounds = sorted(result.evaluate_metrics_clientapp.keys())
        for rnd in rounds:
            # Get evaluation metrics for this round (if available)
            eval_metrics = result.evaluate_metrics_clientapp.get(rnd, {})

            # Get training metrics for this round (if available)
            train_metrics = result.train_metrics_clientapp.get(rnd, {})

            # Write a single row per round
            writer.writerow([
                eval_metrics.get("time", ""),
                eval_metrics.get("eval_loss", ""),
                eval_metrics.get("eval_acc", ""),
                train_metrics.get("train_loss", ""),
            ])

    # Log completion
    log(INFO, f"CSV saved: {path}")