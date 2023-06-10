import numpy as np

"""
Functions that help parse the inputs and the solutions of the optimization problem.
"""

SCHEDULER = 0  # the scheduler used at every hop in the network. 0 for FIFO and 1 for SCED.


def parse_link(network_profile):
    """
    Parse the network profile.
    :param network_profile: a n*m (boolean or integer) matrix that specifies the route of each flow (over nodes).
    :return: a 2-D boolean array describing the route of each flow (over links).
    """
    # Check if the input network is cyclic.
    cyclic = np.issubdtype(network_profile.dtype, np.integer)
    # Initialize the adjacency list.
    links = list()
    for _ in range(network_profile.shape[1]):
        links.append(dict())
    # Fill the adjacency list.
    num_link = 0
    for flow_idx, flow_route in enumerate(network_profile):
        route_nodes = np.where(flow_route)[0]
        # Sort the nodes if the network structure is cyclic.
        if cyclic:
            route_nodes = route_nodes[np.argsort(flow_route[route_nodes])]
        for start, end in zip(route_nodes[:-1], route_nodes[1:]):
            link = links[start].get(end)
            # Create an empty list if link does not exist.
            if link is None:
                link = list()
                links[start][end] = link
                num_link += 1
            # Add flow index to the link.
            link.append(flow_idx)
    # Retrieve the boolean matrix from the adjacency list.
    path_matrix = np.zeros((network_profile.shape[0], num_link), dtype=bool)
    link_idx = 0
    for start_link in links:
        for end_node in start_link:
            path_matrix[:, link_idx][np.array(start_link[end_node])] = True
            link_idx += 1
    return path_matrix


def get_objective(bandwidth, objective, weight):
    """
    Compute the total network bandwidth according to the objective function.
    :param bandwidth: the bandwidth of each link.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: value of the objective bandwidth function.
    """
    if objective == 0:
        total_bandwidth = np.sum(bandwidth)
    elif objective == 1:
        total_bandwidth = np.sum(bandwidth * weight)
    else:
        total_bandwidth = np.amax(bandwidth)
    return total_bandwidth


def parse_solution(path_matrix, solution):
    """
    Parse the NLP solution and return the variables.
    :param path_matrix: the network routes.
    :param solution: solution returned by the NLP solver.
    :return: the variables.
    """
    num_flow, num_link = path_matrix.shape
    reprofiling_delay, bandwidth = np.zeros((num_flow,)), np.zeros((num_link,))
    for flow_idx in range(num_flow):
        reprofiling_delay[flow_idx] = solution[f"D{flow_idx}"]
    for link_idx in range(num_link):
        bandwidth[link_idx] = solution[f"C{link_idx}"]
    ddl = None
    if SCHEDULER == 0:
        ddl = np.zeros((num_link,))
        for link_idx in range(num_link):
            ddl[link_idx] = solution[f"T{link_idx}"]
    elif SCHEDULER == 1:
        ddl = np.zeros_like(path_matrix, dtype=float)
        for flow_idx, flow_route in enumerate(path_matrix):
            for link_idx, flow_link in enumerate(flow_route):
                if flow_link:
                    ddl[flow_idx, link_idx] = solution[f"T{flow_idx}_{link_idx}"]
    return reprofiling_delay, ddl, bandwidth
