import numpy as np
import os
from pathlib import Path

"""Generate flow routes as optimization input."""
# google_nodes = np.zeros((17,), dtype=bool)
# google_nodes[:11] = True
# google_links = [(0, 1), (0, 3), (0, 5), (0, 11), (1, 2), (1, 3), (1, 5), (1, 11), (2, 3), (2, 4), (3, 4), (3, 7),
#                 (3, 12), (4, 7), (4, 10), (4, 13), (5, 6), (5, 7), (5, 8), (5, 12), (5, 13), (6, 8), (6, 9), (6, 14),
#                 (7, 15), (7, 16), (7, 10), (8, 9), (8, 14), (14, 15), (14, 16)]
google_nodes = np.zeros((22,), dtype=bool)
google_nodes[11:] = True
google_links = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (3, 7), (4, 5), (4, 7),
                (4, 10), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (6, 9), (7, 8), (7, 10), (8, 9), (0, 11), (1, 12),
                (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21)]
cev_nodes = np.zeros((44,), dtype=bool)
cev_nodes[13:] = True
cev_links = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (6, 11), (7, 9),
             (7, 11), (8, 10), (9, 12), (0, 13), (0, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (2, 20), (2, 21),
             (2, 22), (3, 23), (3, 24), (4, 25), (5, 26), (5, 27), (6, 28), (6, 29), (6, 30), (7, 31), (7, 32), (7, 33),
             (8, 34), (8, 35), (9, 36), (9, 37), (10, 38), (10, 39), (11, 40), (11, 41), (12, 42), (12, 43)]


def shortest_path_routing(nodes, links, rstate):
    """
    Compute the routing table based on shortest path routing.
    :param nodes: the network nodes.
    :param links: the network links.
    :param rstate: random number generator.
    :return: The shortest route between each source and destination pair.
    """
    # Create the adjacency matrix according to the specified topology.
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for (node1, node2) in links:
        adjacency_matrix[node1, node2] = True
        adjacency_matrix[node2, node1] = True
    # Create the routing table using the shortest paths (minimum hop routing).
    routing_table = np.where(adjacency_matrix, np.arange(len(nodes)), -1)
    hop_count = adjacency_matrix.astype(int)
    for node_idx, mask in enumerate(adjacency_matrix):
        mask = mask.copy()
        # Extract the reachable nodes and randomly shuffle them.
        reachable_nodes = np.arange(len(nodes))[mask]
        rstate.shuffle(reachable_nodes)
        queue = list(reachable_nodes)
        mask[node_idx] = True
        while len(queue):
            node = queue.pop(0)
            # Extract the reachable new nodes and randomly shuffle them.
            new_nodes = np.arange(len(nodes))[adjacency_matrix[node]]
            rstate.shuffle(new_nodes)
            for n in new_nodes:
                if not mask[n]:
                    mask[n] = True
                    routing_table[node_idx, n] = routing_table[node_idx, node]
                    hop_count[node_idx, n] = hop_count[node_idx, node] + 1
                    queue.append(n)
    # Retrieve the route between each source and destination (S-D) pair.
    sd_routes = dict()
    for src_idx in range(len(nodes)):
        for dest_idx in range(len(nodes)):
            if src_idx != dest_idx:
                route = list()
                next_node, table = src_idx, routing_table[:, dest_idx]
                while table[next_node] != -1:
                    route.append(next_node)
                    next_node = table[next_node]
                route.append(next_node)
                sd_routes[(src_idx, dest_idx)] = route
    return sd_routes


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


def generate_random_route(num_flow=None, num_node=None, seed=None, cyclic=False):
    """
    Generate flow routes on a random network.
    :param num_flow: the number of flows in the generated network, randomly selected if None.
    :param num_node: the number of nodes in the generated network, randomly selected if None.
    :param seed: the seed for random generator.
    :param cyclic: whether the graph formed the flow routes is allowed a cyclic structure.
    :return: a numpy matrix describing the routes of the flows.
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
    flow_routes = np.where(mask, route_idx + 1, 0).astype(int) if cyclic else mask
    # Randomly shuffle the nodes traversed by each flow.
    rand_order = np.ones_like(flow_routes, dtype=int)
    for i in range(num_flow):
        rand_order[i] = rstate.permutation(num_node)
    flow_routes = flow_routes[np.arange(num_flow)[:, np.newaxis], rand_order]
    return flow_routes


def generate_tandem_route(num_hop, num_flow_main, num_hop_cross, num_flow_cross, stride_cross=1, pad=True):
    """
    Generate flow routes on the tandem network topology (i.e., parking-lot network).
    Example usage: generate_tandem_route(n, m, 1, 0) for topology 1 (all main flows) with n hops and m main flows.
                   generate_tandem_route(n, m, 2, m//2) for topology 2 (main flows with 2-hop cross flows)
                   with n hops, m main flows, and m cross flows at each hop.
    :param num_hop: the number of hops (links) in the network.
    :param num_flow_main: the number of main flows in the network.
    :param num_hop_cross: the number of hops (links) each cross flow traverses.
    :param num_flow_cross: the number of cross flows entering the network at each entry.
    :param stride_cross: interval (number of hops) between two consecutive cross flow entries.
    :param pad: whether the network extremities are padded.
    :return: a numpy matrix describing the routes of the flows.
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
    flow_routes = np.concatenate((net_main, net_cross), axis=0).astype(bool)
    return flow_routes


def generate_dc_net(net_nodes, net_links, num_pair, source_edge=True, dest_edge=True, routing="shortest_path",
                    routing_path="", prune=True, seed=None):
    """
    Generate flow routes on the inter-datacenter network topology.
    Routes are computed through shortest path.
    :param net_nodes: a boolean array that indicates which nodes are edge nodes.
    :param net_links: a list of links in the network.
    :param num_pair: the number of S-D pairs (and corresponding flow routes) to generate.
    :param source_edge: whether the source node can only be selected from edge nodes.
    :param dest_edge: whether the destination node can only be selected from edge nodes.
    :param routing: the routing protocol. Available choices include "shortest_path" and "custom".
    :param routing_path: path to load the custom routes between S-D pairs. Active only when routing="custom".
    :param prune: Whether to remove the end-hosts from the routes.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the routes of the flows.
    """
    # Initiate random state.
    rstate = np.random.RandomState(seed)
    # Retrieve the routes based on routing protocol.
    if routing == "custom" and os.path.isfile(routing_path):
        route_data = np.load(routing_path)
        routes = dict()
        for route_key in route_data.files:
            route = route_data[route_key]
            routes[(route[0], route[-1])] = route
    else:
        routes = shortest_path_routing(net_nodes, net_links, rstate)
    # Create the flow routes.
    flow_routes = np.zeros((0, len(net_nodes)), dtype=int)
    sd_route = np.zeros((1, len(net_nodes)), dtype=int)
    flow_routes_pruned = np.zeros((0, len(net_nodes) - np.sum(net_nodes)), dtype=int)
    sd_route_pruned = np.zeros((1, len(net_nodes) - np.sum(net_nodes)), dtype=int)
    # Select multiple (num_pair) S-D pairs for the network.
    for _ in range(num_pair):
        sd_route[:], sd_route_pruned[:] = 0, 0
        # Randomly select a source node.
        source = rstate.choice(np.arange(len(net_nodes))[net_nodes]) if source_edge else rstate.randint(len(net_nodes))
        while True:
            # Select a random destination with the path between the corresponding S-D pair covering at least 2 hops.
            destination = rstate.choice(np.arange(len(net_nodes))[net_nodes]) if dest_edge else rstate.randint(
                len(net_nodes))
            if source != destination and len(routes[(source, destination)]) > 2:
                break
        # Establish the path according to the route.
        route = routes[(source, destination)]
        sd_route[0, route] = np.arange(len(route)) + 1
        # Select a random number of flows for each S-D pair.
        # num_flow = rstate.randint(3, 11)
        # net = np.concatenate((net, np.repeat(sd_route, num_flow, axis=0)), axis=0)
        flow_routes = np.concatenate((flow_routes, sd_route), axis=0)
        if prune:
            route_pruned = route[1:-1]
            sd_route_pruned[0, route_pruned] = np.arange(len(route_pruned)) + 1
            flow_routes_pruned = np.concatenate((flow_routes_pruned, sd_route_pruned), axis=0)
    flow_routes, flow_routes_pruned = flow_routes.astype(int), flow_routes_pruned.astype(int)
    return_data = {"routes": flow_routes, "routes_pruned": flow_routes_pruned} if prune else flow_routes
    return return_data


def generate_tsn_net(net_nodes, net_links, num_app, source_edge=True, dest_edge=True, routing="shortest_path",
                     routing_path="", prune=True, seed=None):
    """
    Generate flow routes on the TSN setting.
    Routes are computed through shortest path.
    :param net_nodes: a boolean array that indicates which nodes are end devices.
    :param net_links: a list of links in the network.
    :param num_app: the number of messages to send through unicast, multicast, or broadcast.
    :param source_edge: whether the source node can only be selected from end devices.
    :param dest_edge: whether the destination node can only be selected from end devices.
    :param routing: the routing protocol. Available choices include "shortest_path" and "custom".
    :param routing_path: path to load the custom routes between S-D pairs. Active only when routing="custom".
    :param prune: Whether to remove the end-hosts from the routes.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the routes of the flows.
    """
    # Initiate random state.
    rstate = np.random.RandomState(seed)
    # Retrieve the routes based on routing protocol.
    if routing == "custom" and os.path.isfile(routing_path):
        route_data = np.load(routing_path)
        routes = dict()
        for route_key in route_data.files:
            route = route_data[route_key]
            routes[(route[0], route[-1])] = route
    else:
        routes = shortest_path_routing(net_nodes, net_links, rstate)
    # Create the flow routes.
    flow_routes = np.zeros((0, len(net_nodes)), dtype=int)
    flow_routes_pruned = np.zeros((0, len(net_nodes) - np.sum(net_nodes)), dtype=int)
    # Select multiple (num_app) applications for the network.
    cast_pattern = rstate.choice(3, size=num_app)
    app_dest_num = list()
    for cast in cast_pattern:
        # Randomly select a source node.
        source = rstate.choice(np.arange(len(net_nodes))[net_nodes]) if source_edge else rstate.randint(len(net_nodes))
        # Randomly select destination node(s) according to unicast, multicast, or broadcast.
        dest_mask = net_nodes.copy() if dest_edge else np.ones_like(net_nodes)
        for dest_idx in range(len(net_nodes)):
            if not net_nodes[dest_idx] or source == dest_idx or len(routes[(source, dest_idx)]) <= 2:
                dest_mask[dest_idx] = False
        dest_candidate = np.arange(len(net_nodes))[dest_mask]
        if cast == 0 or len(dest_candidate) == 1:  # Unicast.
            dest_num = 1
            destination = rstate.choice(dest_candidate, size=1)
        elif cast == 1 and len(dest_candidate) > 2:  # Multicast.
            dest_num = rstate.randint(len(dest_candidate) - 2) + 2
            destination = rstate.choice(dest_candidate, size=dest_num, replace=False)
        else:  # Broadcast.
            dest_num = len(dest_candidate)
            destination = dest_candidate
        app_dest_num.append(dest_num)
        # Establish the path according to the route.
        sd_route = np.zeros((dest_num, len(net_nodes)), dtype=int)
        for dest_idx, dest in enumerate(destination):
            route = routes[(source, dest)]
            sd_route[dest_idx, route] = np.arange(len(route)) + 1
        flow_routes = np.concatenate((flow_routes, sd_route), axis=0)
        if prune:
            sd_route_pruned = np.zeros((dest_num, len(net_nodes) - np.sum(net_nodes)), dtype=int)
            for dest_idx, dest in enumerate(destination):
                route_pruned = routes[(source, dest)][1:-1]
                sd_route_pruned[dest_idx, route_pruned] = np.arange(len(route_pruned)) + 1
            flow_routes_pruned = np.concatenate((flow_routes_pruned, sd_route_pruned), axis=0)
    flow_routes, flow_routes_pruned = flow_routes.astype(int), flow_routes_pruned.astype(int)
    return_data = {"routes": flow_routes, "app_dest_num": app_dest_num}
    if prune:
        return_data["routes_pruned"] = flow_routes_pruned
    return return_data


def generate_google_net(num_pair, routing="shortest_path", routing_path="", prune=True, seed=None):
    """
    Generate flow routes using the Google (US) network topology.
    The google network topology is motivated by https://cloud.google.com/about/locations#network.
    """
    return generate_dc_net(google_nodes, google_links, num_pair, source_edge=True, dest_edge=True, routing=routing,
                           routing_path=routing_path, prune=prune, seed=seed)


def generate_chameleon_net(num_pair, k=4, routing="shortest_path", routing_path="", prune=False, seed=None):
    chameleon_nodes, chameleon_links = generate_fat_tree(k)
    """Generate flow routes using the Chameleon fat-tree network topology."""
    return generate_dc_net(chameleon_nodes, chameleon_links, num_pair, source_edge=True, dest_edge=True,
                           routing=routing, routing_path=routing_path, prune=prune, seed=seed)


def generate_cev_net(num_pair, routing="shortest_path", routing_path="", prune=True, seed=None):
    """
    Generate flow routes using the orion CEV network topology.
    The paper is available at https://ieeexplore.ieee.org/abstract/document/8700610/.
    """
    return generate_tsn_net(cev_nodes, cev_links, num_pair, source_edge=True, dest_edge=True, routing=routing,
                            routing_path=routing_path, prune=prune, seed=seed)


def save_file(output_path, file_name, flow_routes):
    """
    Save the generated flow routes to the specified output location.
    :param output_path: the directory to save the flow routes.
    :param file_name: name of the file to save the flow routes.
    :param flow_routes: the flow routes.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if type(flow_routes) is dict:
        np.save(os.path.join(output_path, file_name + ".npz"), **flow_routes)
    else:
        np.save(os.path.join(output_path, file_name + ".npy"), flow_routes)
    return


if __name__ == "__main__":
    # First, specify the directory to save the generated flow routes.
    path = "./network/"
    name = "test"
    # You can specify your own flow routes and directly save it to the directory.
    net = np.array([[1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1]
                    ])
    # save_file(path, name, net.astype(bool))
    # Alternatively, you may generate and save a set of random flow routes.
    # save_file(path, name, generate_random_route(num_flow=3, num_node=5))
    # You may also choose to generate a tandem network.
    # save_file(path, name, generate_tandem_route(10, 2, 2, 1))
    # Or you can generate a network motivated by some realistic network topology.
    # save_file(path, name, generate_google_net(10))  # For the US-Topo (inter-datacenter).
    # save_file(path, name, generate_chameleon_net(10))  # For the fat-tree network (intra-datacenter).
    save_file(path, name, generate_cev_net(10))  # For the Orion CEV network (TSN setting for Ethernet).
