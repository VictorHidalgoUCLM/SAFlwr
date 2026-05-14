"""FedSaSync: A Flower Semi-Asynchronous strategy based on message-based FedAvg aggregation."""
from flwr.server import Grid
import time
from collections.abc import Iterable
from typing import Dict
from flwr.common import Message
import random

def _sample_nodes_semiasync(
    grid: Grid, msg_dict: Dict[str, str], sample_size: int
) -> tuple[list[int], list[int]]:
    """Sample semiasynchornously the specified number of nodes using the Grid.

    Parameters
    ----------
    grid : Grid
        The grid object.
    msg_dict : Dict[str, str]
        A dictionary mapping message IDs to destination node IDs.
    sample_size : int
        The number of nodes to sample.

    Returns
    -------
    tuple[list[int], list[int]]
        A tuple containing the sampled node IDs and the list
        of all connected node IDs.
    """
    all_nodes = list(grid.get_node_ids())   # Get all available nodes in grid
    running_nodes = list(msg_dict.keys())   # Get all nodes that are currently running
    free_nodes = list(set(all_nodes) - set(running_nodes))   

    # Sample nodes that are not currently running
    sampled_nodes = random.sample(
        free_nodes,
        min(len(free_nodes), sample_size)   # Sample only from free nodes, up to the specified sample size
    )
    return sampled_nodes, all_nodes

def _send_and_receive_semiasync(
        grid: Grid,
        messages: Iterable[Message],
        timeout: float | None = None,
        msg_dict: Dict[str, str] = {},
        sync_deg: int = 1,
    ) -> Iterable[Message]:
    """Push messages to specified node IDs and pull semiasynchronously 'M' reply messages.

    This method sends a list of messages to their destination node IDs and then
    waits for 'M' replies. It continues to pull replies until either M replies are
    received or the specified timeout duration is exceeded.
    """
    # Push messages
    msg_ids = grid.push_messages(messages)

    # Register busy nodes in msg_dict
    for msg_id, msg in zip(msg_ids, messages):
        msg_dict[msg.metadata.dst_node_id] = msg_id # node_id: msg_id
    del messages

    # Debug mode: print the msg_dict after pushing messages
    print("msg_dict:", msg_dict)
    
    # Pull messages
    all_msg_ids = set(msg_dict.values())    # Get all message IDs that are currently running
    end_time = time.time() + (timeout if timeout is not None else 0.0)
    ret: list[Message] = []
    while timeout is None or time.time() < end_time:
        res_msgs = grid.pull_messages(all_msg_ids)  # Pull all messages in grid
        ret.extend(res_msgs)
        all_msg_ids.difference_update(
            {msg.metadata.reply_to_message_id for msg in res_msgs}
        )
        # Round end condition: semiaysynchronous round
        # Allow the system to continue if at least M replies have been received
        if len(ret) >= sync_deg:
            break
        # Sleep
        time.sleep(3)

    # Update msg_dict to remove unnecessary entries
    for node_id in list(msg_dict.keys()):
        if msg_dict[node_id] not in all_msg_ids:
            del msg_dict[node_id]

    # Debug mode: print the msg_dict after pulling messages
    print("msg_dict after pulling:", msg_dict)
    return ret