import numpy as np

"""
Functions that help parse the inputs and the solutions of the optimization problem.
"""

SCHEDULER = 0  # the scheduler used at every hop in the network. 0 for FIFO and 1 for SCED.


def parse_link(flow_routes):
    """
    Parse the flow routes.
    :param flow_routes: a n*m (boolean or integer) matrix that specifies the route of each flow (over nodes).
    :return: a 2-D boolean array describing the route of each flow (over links).
    """
    # Check if the input network is cyclic.
    cyclic = np.issubdtype(flow_routes.dtype, np.integer)
    # Initialize the adjacency list.
    links = list()
    flow_links = list()
    for _ in range(flow_routes.shape[1]):
        links.append(dict())
    # Fill the adjacency list.
    num_link = 0
    for flow_idx, flow_route in enumerate(flow_routes):
        flow_link = list()
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
            flow_link.append((start, end))
        flow_links.append(flow_link)
    # Retrieve the boolean matrix from the adjacency list.
    path_matrix = np.zeros((flow_routes.shape[0], num_link), dtype=bool)
    link_map, link_idx = list(), 0
    for start_node, start_link in enumerate(links):
        for end_node in start_link:
            path_matrix[:, link_idx][np.array(start_link[end_node])] = True
            link_map.append((start_node, end_node))
            link_idx += 1
    # Restore the order that each flow traverses the links.
    path_order = np.zeros_like(path_matrix, dtype=int) - 1
    for flow_idx, flow_link in enumerate(flow_links):
        for link_idx, link in enumerate(flow_link):
            path_order[flow_idx, link_map.index(link)] = link_idx
    return path_matrix, path_order, link_map


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


def get_buffer_bound(path_matrix, path_order, flow_profile, reprofiling_delay, ddl, bandwidth):
    """
    Compute the buffer bounds for each link and each reprofiler assuming a per-flow reprofiling and
    non-work-conserving setting.
    :param path_matrix: the network routes.
    :param path_order: the order by which each flow traverses the links.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the link delays of the solution.
    :param bandwidth: the link bandwidth.
    :return: the per-link and per-reprofiler buffer bounds.
    """
    zero_ddl = 1e-15
    num_flow, num_link = path_matrix.shape
    link_bound = np.zeros((num_link,))
    reprofiler_bound = np.zeros_like(path_matrix, dtype=float)
    # Check the scheduling policy.
    if SCHEDULER == 0:
        ddl = np.ones((num_flow, 1)) * ddl
        ddl = np.where(path_matrix, ddl, 0)
    zs_mask = reprofiling_delay < zero_ddl
    reprofiling_delay[zs_mask] = 0
    short_rate = np.divide(flow_profile[:, 1], reprofiling_delay, out=np.copy(flow_profile[:, 0]),
                           where=np.logical_not(zs_mask))
    burst = np.where(zs_mask, flow_profile[:, 1], 0)
    rate = np.concatenate((short_rate, flow_profile[:, 0] - short_rate))
    for link_idx in range(num_link):
        link_bound[link_idx] = link_buffer(path_matrix[:, link_idx], rate, burst, reprofiling_delay,
                                           bandwidth[link_idx])
    for flow_idx in range(num_flow):
        reprofiler_bound[flow_idx] = reprofiler_buffer(path_matrix[flow_idx], path_order[flow_idx], ddl[flow_idx],
                                                       flow_profile[flow_idx, 0], flow_profile[flow_idx, 1],
                                                       reprofiling_delay[flow_idx])
    return link_bound, reprofiler_bound


def link_buffer(link_mask, rate, burst, reprofiling_delay, bandwidth):
    """
    Calculate the buffer bound of one link.
    :param link_mask: mask to retrieve the subset of flows at this hop.
    :param rate: the short-term and long-term rates of the flows.
    :param burst: the burst sizes of the flows.
    :param reprofiling_delay: the reprofiling delays of the flows.
    :param bandwidth: the link bandwidth.
    :return: the buffer bound.
    """
    num_flow = len(link_mask)
    link_burst = np.sum(burst[link_mask])
    # Only retrieve flows with a non-zero reprofiling delay.
    reprofiling_mask = reprofiling_delay > 0
    link_mask = np.logical_and(link_mask, reprofiling_mask)
    short_rate = np.sum(rate[:num_flow][link_mask])
    link_sort = np.argsort(reprofiling_delay)
    link_mask = link_mask[link_sort]
    link_sort = link_sort[link_mask]
    link_rp, link_rate = reprofiling_delay[link_sort], rate[num_flow:][link_sort]
    rate_cum = np.cumsum(np.append(short_rate, link_rate))
    link_rp_ = np.append(0, link_rp)
    rp_int = link_rp_[1:] - link_rp_[:-1]
    # Compute the values on each inflection point of the arrival curve.
    arrival_curve = rate_cum[:-1] * rp_int
    arrival_curve = np.cumsum(arrival_curve)
    arrival_curve = np.append(0, arrival_curve)
    # Identify the point where the buffer upper bound is achieved.
    buffer_idx = np.argmax(rate_cum <= bandwidth)
    buffer_bound = (arrival_curve[buffer_idx] + link_burst) - (bandwidth * link_rp_[buffer_idx])
    return buffer_bound


def reprofiler_buffer(path_mask, path_order, link_ddl, rate, burst, reprofiling_delay):
    """
    Calculate the buffer bounds of the reprofilers across all the hops a flow traverses.
    :param path_mask: mask to retrieve the subset of links that the flow traverses.
    :param path_order: the order that the links are traversed.
    :param link_ddl: the local deadline of the flow at each hop.
    :param rate: the token rate of the flow.
    :param burst: the burst size of the flow.
    :param reprofiling_delay: the reprofiling delay of the flow.
    :return: the buffer bounds.
    """
    # Retrieve all the local deadlines up to the last hop.
    buffer_bound = np.zeros_like(link_ddl)
    link_ddl = link_ddl[path_mask]
    link_order = path_order[path_mask]
    link_ddl[link_order] = link_ddl
    link_ddl = link_ddl[:-1]

    # Helper function to compute value on the reprofiler.
    def reprofile(x):
        if x < reprofiling_delay:
            return x * (burst / reprofiling_delay)
        else:
            return burst + (x - reprofiling_delay) * rate

    buffer = np.array(list(map(reprofile, link_ddl)))
    # The buffer bound at the first hop is simply the original burst size.
    buffer = np.append(burst, buffer)
    buffer_bound[path_mask] = buffer[link_order]
    return buffer_bound


def get_density_score(path_matrix):
    """
    Compute the density score of a flow path matrix. Generally, a higher density score indicates a higher percentage of
    link coverage of the flows.
    :param path_matrix: the network routes.
    :return: the density score of the network.
    """
    # score = np.sum(path_matrix) / np.size(path_matrix)
    _, num_link = path_matrix.shape
    num_flow_per_hop, score = np.sum(path_matrix, axis=0), 0
    if num_link > 1:
        for link_idx in range(num_link):
            sub_net = path_matrix[path_matrix[:, link_idx]]
            score += np.sum(np.sum(sub_net, axis=0) / num_flow_per_hop) - 1
        score /= (num_link - 1) * num_link
    return score
