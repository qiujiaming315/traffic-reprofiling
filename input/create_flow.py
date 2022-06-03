import numpy as np
import os
from pathlib import Path

"""Generate (random) flow profile as input file for optimization."""


def generate_random_flow(num_flow=None, seed=None):
    """
    Generate random flow profiles.
    :param num_flow: the number of flows in the generated profile, randomly selected if None.
    :param seed: the random seed for network generation.
    :return: a numpy matrix that describes the flow profile.
    """
    # Set some parameters.
    flow_bound = (2, 11)
    # rate_bound = (0, 5)
    # burst_bound = (0, 20)
    # deadline_class = np.array([1, 5, 10, 20])
    rate_bound = (1, 100)
    burst_bound = (1, 100)
    deadline_class = np.array([0.01, 0.1, 1])
    # deadline_class = np.array([0.01, 0.025, 0.05, 0.1])
    # Randomly generate a flow profile.
    rstate = np.random.RandomState(seed)
    if num_flow is None:
        num_flow = rstate.randint(flow_bound[0], flow_bound[1])
    flow = np.zeros((num_flow, 3))
    rand_data = rstate.rand(num_flow, 2)
    flow[:, 0] = np.around(rand_data[:, 0] * (rate_bound[1] - rate_bound[0]) + rate_bound[0], 2)
    flow[:, 1] = np.around(rand_data[:, 1] * (burst_bound[1] - burst_bound[0]) + burst_bound[0], 2)
    flow[:, 2] = deadline_class[rstate.randint(len(deadline_class), size=num_flow)]
    return flow


def save_file(output_path, flow, per_hop=False):
    """
    Save the generated network routes to the specified output location.
    :param output_path: the path of directory to save the flow profile.
    :param flow: the flow profile.
    :param per_hop: whether the deadline stands for per hop deadline (True) or end-to-end deadline (False).
    """
    num_flow = flow.shape[0]
    output_path = os.path.join(output_path, str(num_flow), "")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.savez(f"{output_path}flow{num_files + 1}.npz", flow=flow, per_hop=per_hop)
    return


if __name__ == "__main__":
    output_path = "./flow/parking_lot/"
    flow = np.array([[15.0, 8.0, 1.0],
                     [2.0, 4.0, 2.0],
                     [3.0, 13.0, 5.0]
                     ])
    for _ in range(1000):
        save_file(output_path, generate_random_flow(23), True)
    # save_file(output_path, flow)
