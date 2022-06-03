import numpy as np
import os
from pathlib import Path

"""Generate (random) network topology and route as input file for optimization."""


def generate_random_net(num_flow=None, num_node=None, seed=None, cyclic=False, dist='uniform'):
    """
    Generate random networks.
    :param num_node: the number of flows in the generated network, randomly selected if None.
    :param num_flow: the number of nodes in the generated network, randomly selected if None.
    :param seed: the random seed for network generation.
    :param cyclic: whether the network is cyclic.
    :param dist: distribution that the number of nodes each flow passes is sampled from.
    :return: a numpy matrix that describes the network routing for each flow.
    """
    # TODO: The current implementation of random cyclic network is highly sparse. Add a handler to control sparsity.
    rstate = np.random.RandomState(seed)
    net_size = rstate.randint(2, 11, size=2)
    if num_flow is None:
        num_flow = net_size[0]
    if num_node is None:
        num_node = net_size[1]
    if dist == 'uniform':
        flow_node = rstate.randint(2, num_node + 1, size=(num_flow, 1))
    route_idx = np.arange(num_node) * np.ones((num_flow, 1))
    mask = route_idx < flow_node
    net = np.where(mask, route_idx + 1, 0).astype(int) if cyclic else mask
    rand_order = np.ones_like(net, dtype=int)
    for i in range(num_flow):
        rand_order[i] = rstate.permutation(num_node)
        if seed is not None:
            seed *= 3
            rstate.seed((seed + 5) % 2 ** 32)
    net = net[np.arange(num_flow)[:, np.newaxis], rand_order]
    return net


def generate_parking_lot(num_hop, num_flow_main, num_hop_cross, num_flow_cross, stride_cross=1, pad=True):
    """
    Generate a network with parking lot topology (i.e., tandem network).
    :param num_hop: the number of hops in the generated network.
    :param num_flow_main: the number of main flows in the generated network.
    :param num_hop_cross: the number of hop each cross flow passes.
    :param num_flow_cross: the number of cross flows that enter the network at each entry.
    :param stride_cross: the stride of the cross flow entries.
    :param pad: whether the cross traffic are padded.
    :return: a numpy matrix that describes the network routing for each flow.
    """
    net_main = np.ones((num_flow_main, num_hop + 1), dtype=int)
    if num_hop >= num_hop_cross:
        num_col = num_hop + 1
        net_cross1 = np.eye(num_hop - num_hop_cross + 1, num_col, dtype=int)
        net_cross2 = np.eye(num_hop - num_hop_cross + 1, num_col, k=num_hop_cross + 1, dtype=int)
        net_cross = np.cumsum(net_cross1 - net_cross2, axis=1)
    else:
        num_col = num_hop_cross
        net_cross = np.ones((0, num_col), dtype=int)
    if pad:
        net_pad = np.eye(num_hop_cross - 1, num_col, k=num_col - num_hop_cross, dtype=int)
        net_pad = np.cumsum(net_pad, axis=1)
        net_cross = np.concatenate((net_cross, net_pad), axis=0)
        net_cross = np.concatenate((net_pad[::-1, ::-1], net_cross), axis=0)
    net_cross = net_cross[::stride_cross]
    net_cross = np.repeat(net_cross, num_flow_cross, axis=0)
    net_cross = net_cross[:, :num_hop + 1]
    net_cross = net_cross[np.sum(net_cross, axis=1) > 1]
    net = np.concatenate((net_main, net_cross), axis=0).astype(bool)
    return net


def save_file(output_path, net):
    """
    Save the generated network routes to the specified output location.
    :param output_path: the path of directory to save the network profile.
    :param net: the network profile.
    """
    num_flow = net.shape[0]
    output_path = os.path.join(output_path, str(num_flow), "")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.save(f"{output_path}net{num_files + 1}.npy", net)
    return


if __name__ == "__main__":
    output_path = "./network/feed_forward/parking_lot/"
    net = np.array([[1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1]
                    ])
    save_file(output_path, generate_parking_lot(2, 4, 4, 1, 1))
    # save_file(output_path, generate_parking_lot(9, 3, 3, 2, 3))
    # save_file(output_path, net.astype(bool))
