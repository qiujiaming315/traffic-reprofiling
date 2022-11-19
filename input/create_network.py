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


def generate_fat_tree(k=4):
    """
    Generate the fat-tree data center network topology.
    :param k: the number of pods.
    :return: an boolean array and a list of tuples that describes the nodes and links in the network, respectively.
    """
    # Sanity check on the parameter k.
    assert k % 2 == 0
    # Define the nodes in the network
    net_nodes = np.arange(k ** 2 * 5 // 4 + k ** 3 // 4)
    agg_nodes = np.logical_and(net_nodes >= k ** 2 // 4, net_nodes < k ** 2 * 3 // 4)
    edge_nodes = np.logical_and(net_nodes >= k ** 2 * 3 // 4, net_nodes < k ** 2 * 5 // 4)
    server_nodes = net_nodes >= k ** 2 * 5 // 4
    # Set the links in the network
    net_links = []
    # Set the links between core and aggregation nodes.
    for agg_idx in net_nodes[agg_nodes]:
        agg_idx_ = agg_idx - k ** 2 // 4
        agg_pod_idx = agg_idx_ % (k // 2)
        for i in range(k // 2):
            net_links.append((agg_pod_idx * k // 2 + i, agg_idx))
    # Set the links between aggregation and edge nodes.
    for agg_idx in net_nodes[agg_nodes]:
        agg_idx_ = agg_idx - k ** 2 // 4
        pod_idx = agg_idx_ // (k // 2)
        for i in range(k // 2):
            net_links.append((agg_idx, pod_idx * k // 2 + i + k ** 2 * 3 // 4))
    # Set the links between edge and server nodes.
    for edge_idx in net_nodes[edge_nodes]:
        edge_idx_ = edge_idx - k ** 2 * 3 // 4
        pod_idx, edge_pod_idx = divmod(edge_idx_, (k // 2))
        for i in range(k // 2):
            net_links.append((edge_idx, pod_idx * k ** 2 // 4 + edge_pod_idx * k // 2 + i + k ** 2 * 5 // 4))
    return server_nodes, net_links


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


def generate_realistic_net(net_nodes, net_links, num_pair, source_edge=True, dest_edge=True, seed=None):
    """
    Generate a network profile using the some realistic (or realistic motivated) network topology.
    :param net_nodes: a boolean array that indicates which nodes are edge nodes
                      (all the edge nodes should have smaller indices).
    :param net_links: a list of links in the network.
    :param num_pair: the number of S-D pairs in the network profile.
    :param seed: the seed for random generator.
    :param source_edge: whether the source node can only be selected from edge nodes.
    :param dest_edge: whether the destination node can only be selected from edge nodes.
    :return: a numpy matrix describing the routes of the flows (network profile).
    """
    # Initiate random state and define a function for renewing the random seed (deterministically).
    rstate = np.random.RandomState(seed)

    def renew_seed(s):
        if s is not None:
            s *= 3
            rstate.seed((s + 5) % 2 ** 32)
        return s

    # Create the adjacency matrix according to the specified topology.
    adjacency_matrix = np.zeros((len(net_nodes), len(net_nodes)), dtype=bool)
    for (node1, node2) in net_links:
        adjacency_matrix[node1, node2] = True
        adjacency_matrix[node2, node1] = True
    # Create the routing table using the shortest paths (minimum hop routing).
    routing_table = np.where(adjacency_matrix, np.arange(len(net_nodes)), -1)
    distance = adjacency_matrix.astype(int)
    for node_idx, mask in enumerate(adjacency_matrix):
        mask = mask.copy()
        # Extract the reachable nodes and randomly shuffle them.
        reachable_nodes = np.arange(len(net_nodes))[mask]
        rstate.shuffle(reachable_nodes)
        seed = renew_seed(seed)
        queue = list(reachable_nodes)
        mask[node_idx] = True
        while len(queue):
            node = queue.pop(0)
            # Extract the reachable new nodes and randomly shuffle them.
            new_nodes = np.arange(len(net_nodes))[adjacency_matrix[node]]
            rstate.shuffle(new_nodes)
            seed = renew_seed(seed)
            for n in new_nodes:
                if not mask[n]:
                    mask[n] = True
                    routing_table[node_idx, n] = routing_table[node_idx, node]
                    distance[node_idx, n] = distance[node_idx, node] + 1
                    queue.append(n)
    # Create the network profile.
    net = np.zeros((0, len(net_nodes)), dtype=int)
    # Select multiple (num_pair) S-D pairs for the network.
    for _ in range(num_pair):
        sd_route = np.zeros((1, len(net_nodes)), dtype=int)
        # Randomly select a source node.
        source = rstate.choice(np.arange(len(net_nodes))[net_nodes]) if source_edge else rstate.randint(len(net_nodes))
        destination = source
        while distance[source, destination] < 2:
            # Generate a new seed.
            seed = renew_seed(seed)
            # Select a random destination with the path between the corresponding S-D pair covering at least 2 hops.
            destination = rstate.choice(np.arange(len(net_nodes))[net_nodes]) if dest_edge else rstate.randint(
                len(net_nodes))
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
        seed = renew_seed(seed)
    return net.astype(int)


def generate_google_net(num_pair, seed=None):
    """
    Generate a network profile using the Google (US) network topology.
    The google network topology is motivated by https://cloud.google.com/about/locations#network.
    """
    return generate_realistic_net(google_nodes, google_links, num_pair, True, False, seed)


def generate_chameleon_net(num_pair, k=4, seed=None):
    chameleon_nodes, chameleon_links = generate_fat_tree(k)
    """Generate a network profile using the Chameleon fat-tree network topology."""
    return generate_realistic_net(chameleon_nodes, chameleon_links, num_pair, True, True, seed)


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
    # First, specify the directory to save the generated network profiles.
    path = "./network/"
    # path = "./network/practical/fat_tree/k4/"
    # You can specify your own network profile and directly save it to the directory.
    net = np.array([[1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1]
                    ])
    save_file(path, net.astype(bool))
    # Alternatively, you may generate and save a random network profile.
    save_file(path, generate_random_net(num_flow=3, num_node=5))
    # You may also choose to generate a tandem network.
    save_file(path, generate_tandem_net(10, 2, 2, 1))
    # Or you can generate a network motivated by some realistic network topology.
    save_file(path, generate_google_net(10))
    save_file(path, generate_chameleon_net(10))
    # for num_sd in range(10, 251, 10):
    #     for _ in range(20):
    #         save_file(path + f"{num_sd}/", generate_chameleon_net(num_sd))
