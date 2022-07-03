import numpy as np
import os
from pathlib import Path

"""Generate network profiles (topology + flow routes) as optimization input."""
# google_nodes = np.zeros((17,), dtype=bool)
# google_nodes[:11] = True
# google_links = [(0, 1), (0, 3), (0, 5), (0, 11), (1, 2), (1, 3), (1, 5), (1, 11), (2, 3), (2, 4), (3, 4), (3, 7),
#                 (3, 12), (4, 7), (4, 10), (4, 13), (5, 6), (5, 7), (5, 8), (5, 12), (5, 13), (6, 8), (6, 9), (6, 14),
#                 (7, 15), (7, 16), (7, 10), (8, 9), (8, 14), (14, 15), (14, 16)]
google_nodes = np.ones((11,), dtype=bool)
google_links = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (3, 7), (4, 5), (4, 7),
                (4, 10), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (6, 9), (7, 8), (7, 10), (8, 9)]


def generate_random_net(num_flow=None, num_node=None, seed=None, cyclic=False):
    """
    Generate random network profiles.
    :param num_flow: the number of flows in the generated network, randomly selected if None.
    :param num_node: the number of nodes in the generated network, randomly selected if None.
    :param seed: the seed for random generator.
    :param cyclic: whether the graph formed the flow routes is allowed a cyclic structure.
    :return: a numpy matrix describing the routes of the flows (network profile).
    """
    rstate = np.random.RandomState(seed)
    net_size = rstate.randint(2, 11, size=2)
    if num_flow is None:
        num_flow = net_size[0]
    if num_node is None:
        num_node = net_size[1]
    # Randomly select the number of nodes each flow traverse.
    flow_node = rstate.randint(2, num_node + 1, size=(num_flow, 1))
    route_idx = np.arange(num_node) * np.ones((num_flow, 1))
    mask = route_idx < flow_node
    net = np.where(mask, route_idx + 1, 0).astype(int) if cyclic else mask
    # Randomly shuffle the nodes traversed by each flow to create an arbitrary network profile.
    rand_order = np.ones_like(net, dtype=int)
    for i in range(num_flow):
        rand_order[i] = rstate.permutation(num_node)
        # Generate a new seed (deterministically) to shuffle the nodes of each flow differently.
        if seed is not None:
            seed *= 3
            rstate.seed((seed + 5) % 2 ** 32)
    net = net[np.arange(num_flow)[:, np.newaxis], rand_order]
    return net


def generate_tandem_net(num_hop, num_flow_main, num_hop_cross, num_flow_cross, stride_cross=1, pad=True):
    """
    Generate a network profile with tandem network topology (i.e., parking-lot network).
    Example usage: generate_tandem_net(n, m, 1, 0) for network topology 1 (all main flows) with n hops and m main flows.
                   generate_tandem_net(n, m, 2, m//2) for network topology 2 (main flows with 2-hop cross flows)
                   with n hops, m main flows, and m cross flows at each hop.
    :param num_hop: the number of hops (links) in the generated network.
    :param num_flow_main: the number of main flows in the generated network.
    :param num_hop_cross: the number of hops (links) each cross flow traverses.
    :param num_flow_cross: the number of cross flows entering the network at each entry.
    :param stride_cross: interval (number of hops) between two consecutive cross flow entries.
    :param pad: whether the network extremities are padded.
    :return: a numpy matrix describing the routes of the flows (network profile).
    """
    # Create the main flows in the network.
    net_main = np.ones((num_flow_main, num_hop + 1), dtype=int)
    # Create cross flows in the network.
    if num_hop >= num_hop_cross:
        num_col = num_hop + 1
        net_cross1 = np.eye(num_hop - num_hop_cross + 1, num_col, dtype=int)
        net_cross2 = np.eye(num_hop - num_hop_cross + 1, num_col, k=num_hop_cross + 1, dtype=int)
        net_cross = np.cumsum(net_cross1 - net_cross2, axis=1)
    else:
        num_col = num_hop_cross
        net_cross = np.ones((0, num_col), dtype=int)
    # Pad the network extremities with additional cross flows.
    if pad:
        net_pad = np.eye(num_hop_cross - 1, num_col, k=num_col - num_hop_cross, dtype=int)
        net_pad = np.cumsum(net_pad, axis=1)
        net_cross = np.concatenate((net_cross, net_pad), axis=0)
        net_cross = np.concatenate((net_pad[::-1, ::-1], net_cross), axis=0)
    # Select a subset of cross flows according to the interval (stride_cross) between entries.
    net_cross = net_cross[::stride_cross]
    # Create multiple (num_flow_cross) cross flows at each entry.
    net_cross = np.repeat(net_cross, num_flow_cross, axis=0)
    # Clip the cross flows.
    net_cross = net_cross[:, :num_hop + 1]
    # Abandon cross flows that traverses less than 2 nodes (0 hop).
    net_cross = net_cross[np.sum(net_cross, axis=1) > 1]
    # Combine the main and cross flows.
    net = np.concatenate((net_main, net_cross), axis=0).astype(bool)
    return net


def generate_google_net(num_pair, seed=None):
    """
    Generate a network profile using the Google (US) network topology.
    :param num_pair: the number of S-D pairs in the network profile.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the routes of the flows (network profile).
    """
    # Create the adjacency matrix according to the Google topology.
    google_matrix = np.zeros((len(google_nodes), len(google_nodes)), dtype=bool)
    for (node1, node2) in google_links:
        google_matrix[node1, node2] = True
        google_matrix[node2, node1] = True
    # Create the routing table using the shortest paths (minimum hop routing).
    routing_table = np.where(google_matrix, np.arange(len(google_nodes)), -1)
    distance = google_matrix.astype(int)
    for node_idx, mask in enumerate(google_matrix):
        mask = mask.copy()
        queue = list(np.arange(len(google_nodes))[mask])
        mask[node_idx] = True
        while len(queue):
            node = queue.pop(0)
            new_nodes = np.arange(len(google_nodes))[google_matrix[node]]
            for n in new_nodes:
                if not mask[n]:
                    mask[n] = True
                    routing_table[node_idx, n] = routing_table[node_idx, node]
                    distance[node_idx, n] = distance[node_idx, node] + 1
                    queue.append(n)
    # Create the network profile.
    rstate = np.random.RandomState(seed)
    net = np.zeros((0, len(google_nodes)), dtype=int)
    # Select multiple (num_pair) S-D pairs for the network.
    for _ in range(num_pair):
        sd_route = np.zeros((1, len(google_nodes)), dtype=int)
        source = rstate.randint(np.sum(google_nodes))
        destination = source
        while distance[source, destination] < 2:
            # Select a random destination with the path between the corresponding S-D pair covering at least 2 hops.
            destination = rstate.randint(len(google_nodes))
        # Establish the path according to the routing table.
        idx, node, table = 1, source, routing_table[:, destination]
        while table[node] != -1:
            sd_route[0, node] = idx
            idx += 1
            node = table[node]
        sd_route[0, node] = idx
        # Select a random number of flows for each S-D pair.
        num_flow = rstate.randint(3, 11)
        net = np.concatenate((net, np.repeat(sd_route, num_flow, axis=0)), axis=0)
        # Generate a new seed.
        if seed is not None:
            seed *= 3
            rstate.seed((seed + 5) % 2 ** 32)
    return net.astype(int)


def save_file(output_path, net):
    """
    Save the generated network profile to the specified output location.
    :param output_path: the directory to save the network profile.
    :param net: the network profile.
    """
    num_flow = net.shape[0]
    output_path = os.path.join(output_path, str(num_flow), "")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.save(f"{output_path}net{num_files + 1}.npy", net)
    return


if __name__ == "__main__":
    path = "./network/"
    save_file(path, generate_google_net(10))
