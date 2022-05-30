import numpy as np
import bisect


def parse_link(network_profile):
    """
    Parse the network profile.
    :param network_profile: a n*m (boolean or integer) matrix that specifies the route of each flow (over nodes).
    :return: a 2-D boolean array describing the route of each flow (over links).
    """
    # Check if the input network is cyclic.
    cyclic = network_profile.dtype is np.dtype(int)
    # Initialize the adjacency list.
    links = list()
    for _ in range(network_profile.shape[1]):
        links.append(dict())
    # Fill the adjacency list.
    num_link = 0
    for flow_idx, route in enumerate(network_profile):
        route_nodes = np.where(route)[0]
        # Sort the nodes if the network structure is cyclic.
        if cyclic:
            route_nodes = route_nodes[np.argsort(route[route_nodes])]
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
    route = np.zeros((network_profile.shape[0], num_link), dtype=bool)
    link_idx = 0
    for start_link in links:
        for end_node in start_link:
            route[:, link_idx][np.array(start_link[end_node])] = True
            link_idx += 1
    return route


def rate_proportional(route, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by the rate proportional solution.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the minimum bandwidth.
    """
    flow_rate = np.concatenate(
        (flow_profile[:, 0][np.newaxis, :], flow_profile[:, 1][np.newaxis, :] / flow_profile[:, 2][np.newaxis, :]),
        axis=0)
    flow_rate = np.amax(flow_rate, axis=0)
    bandwidth = np.sum(route * flow_rate[:, np.newaxis], axis=0)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth


def no_shaping(route, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by no shaping and evenly split local deadlines.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the minimum bandwidth.
    """
    num_flow, num_link = route.shape
    shaping_delay = np.zeros((num_flow,))
    ddl = (flow_profile[:, 2] / np.sum(route, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(route, ddl, 0)
    bandwidth = bandwidth_one_slope(route, flow_profile, shaping_delay, ddl)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth


def greedy(route, flow_profile, objective, weight, num_iter):
    """
    Use the heuristic-based greedy algorithm to efficiently approximate the optimal solution (for two-slope shapers
    only, guaranteed to be no worse than both the rate-proportional solution and the no shaping solution).
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :param num_iter: number of iteration.
    :return: the improved solution.
    """
    # Compute the rate required by each flow under rate-proportional solution.
    zero_ddl = 1e-5
    num_flow, num_link = route.shape
    flow_rate = np.concatenate(
        (flow_profile[:, 0][np.newaxis, :], flow_profile[:, 1][np.newaxis, :] / flow_profile[:, 2][np.newaxis, :]),
        axis=0)
    flow_rate = np.amax(flow_rate, axis=0)
    best_solution = get_objective(np.sum(route * flow_rate[:, np.newaxis], axis=0), objective, weight)
    burst, deadline = flow_profile[:, 1], flow_profile[:, 2]
    # Transform the rates into angles and create evenly split angles to generate different initial solutions.
    flow_arc = np.arctan(flow_rate)
    flow_arcs = np.linspace(flow_arc, np.pi / 2, num=num_iter + 1)
    for arc in flow_arcs[1:]:
        rate = np.tan(arc)
        shaping_delay = burst / rate
        # Handle the special case of no shaping.
        if all(shaping_delay < zero_ddl):
            shaping_delay = np.zeros_like(shaping_delay)
        ddl = ((deadline - shaping_delay) / np.sum(route, axis=1))[:, np.newaxis] * np.ones((num_link,))
        ddl = np.where(route, ddl, 0)
        bandwidth = bandwidth_two_slope(route, flow_profile, shaping_delay, ddl)
        new_solution = improve_solution(route, flow_profile, (shaping_delay, ddl, bandwidth), objective, weight, True)
        best_solution = min(best_solution, new_solution)
    return best_solution


def get_objective(bandwidth, objective, weight):
    """
    Compute the network bandwidth according to the objective function.
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


"""
The following utility functions can be used for multiple supplementary purposes
and are not actively used in this implementation.
"""


def check_solution(route, flow_profile, solution, two_slope=False):
    """
    Check whether the actual bandwidth is consistent with the solution result.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param solution: solution returned by the NLP solver.
    :param two_slope: whether the solution uses two slope shapers.
    :return: consistency of the solution.
    """
    err = 1e-3
    shaping_delay, ddl, bandwidth = parse_solution(route, solution)
    # Check if solution deadlines are non-negative.
    feasible1 = np.all(shaping_delay >= 0) and np.all(ddl >= 0)
    # Check if each flow in-network deadline stays in range.
    total_ddl, sd_ub = flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0]
    net_ddl = np.sum(ddl, axis=1)
    feasible2 = np.all(net_ddl + shaping_delay <= total_ddl + err)
    feasible3 = np.all(shaping_delay <= sd_ub + err)
    feasible = feasible1 and feasible2 and feasible3
    if two_slope:
        actual_bandwidth = bandwidth_two_slope(route, flow_profile, shaping_delay, ddl)
    else:
        actual_bandwidth = bandwidth_one_slope(route, flow_profile, shaping_delay, ddl)
    tight = np.all(np.abs(actual_bandwidth - bandwidth) <= err)
    return feasible and tight


def parse_solution(route, solution):
    """
    Parse the NLP solution and return the variables.
    :param route: the network routes.
    :param solution: solution returned by the NLP solver.
    :return: the variables.
    """
    num_flow, num_link = route.shape
    shaping_delay, bandwidth = np.zeros((num_flow,)), np.zeros((num_link,))
    ddl = np.zeros_like(route, dtype=float)
    for flow_idx in range(num_flow):
        shaping_delay[flow_idx] = solution[f"s{flow_idx}"]
    for link_idx in range(num_link):
        bandwidth[link_idx] = solution[f"B{link_idx}"]
    for flow_idx, flow_route in enumerate(route):
        for link_idx, flow_link in enumerate(flow_route):
            if flow_link:
                ddl[flow_idx, link_idx] = solution[f"d{flow_idx}_{link_idx}"]
    return shaping_delay, ddl, bandwidth


def improve_solution(route, flow_profile, solution, objective, weight, two_slope=False):
    """
    Improve the solution using a heuristic-based iterative algorithm.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param solution: solution returned by the NLP solver.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :param two_slope: whether the solution uses two slope shapers.
    :return: the improved solution.
    """
    # Make sure the solution passes the sanity check.
    err = 1e-3
    if type(solution) is dict:
        check_solution(route, flow_profile, solution, two_slope)
        shaping_delay, ddl, bandwidth = parse_solution(route, solution)
    else:
        shaping_delay, ddl, bandwidth = solution
    improve_func = improve_two_slope if two_slope else improve_one_slope
    prev_bandwidth = np.inf
    # Run the algorithm iteratively until the solution cannot be further improved.
    while True:
        current_bandwidth = get_objective(bandwidth, objective, weight)
        # Break when the improvement is marginal.
        if prev_bandwidth - current_bandwidth < err:
            break
        prev_bandwidth = current_bandwidth
        # Improve the solution.
        shaping_delay, ddl, bandwidth = improve_func(route, flow_profile, shaping_delay, ddl)
    return current_bandwidth


def bandwidth_one_slope(route, flow_profile, shaping_delay, ddl):
    """
    Calculate the actual bandwidth (with one-slope shapers) according to the solution variables.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param shaping_delay: the shaping delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the actual per-hop bandwidth.
    """
    num_link = route.shape[1]
    actual_bandwidth = np.zeros((num_link,))
    rate, burst = flow_profile[:, 0], flow_profile[:, 1] - (flow_profile[:, 0] * shaping_delay)
    for link_idx in range(num_link):
        min_bd, link_rate = bandwidth_one_slope_(route[:, link_idx], ddl[:, link_idx], rate, burst)
        actual_bandwidth[link_idx] = max(np.nanmax(min_bd), link_rate)
    return actual_bandwidth


def bandwidth_one_slope_(link_mask, link_ddl, rate, burst):
    """
    Calculate the actual bandwidth (with one-slope shapers) at one hop.
    :param link_mask: mask to retrieve the subset of flows at this hop.
    :param link_ddl: the local deadlines of the flows at this hop.
    :param rate: the long-term rates of the flows.
    :param burst: the burst sizes of the flows.
    :return: the bandwidth requirement at the inflection points of the aggregate service curve,
             and the aggregate long-term rate.
    """
    zero_ddl = 1e-5
    link_sort = np.argsort(link_ddl)
    link_mask = link_mask[link_sort]
    link_sort = link_sort[link_mask]
    link_ddl, link_rate, link_burst = link_ddl[link_sort], rate[link_sort], burst[link_sort]
    rate_cum = np.cumsum(np.append(0, link_rate)[:-1])
    link_ddl_ = np.append(0, link_ddl)
    ddl_int = link_ddl_[1:] - link_ddl_[:-1]
    min_bd = rate_cum * ddl_int + link_burst
    min_bd = np.cumsum(min_bd)
    zero_mask = np.logical_and(min_bd < zero_ddl, link_ddl < zero_ddl)
    min_bd = min_bd / link_ddl
    min_bd = np.where(zero_mask, 0, min_bd)
    return min_bd, np.sum(link_rate)


def bandwidth_two_slope(route, flow_profile, shaping_delay, ddl):
    """
    Calculate the actual bandwidth (with two-slope shapers) according to the solution variables.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param shaping_delay: the shaping delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the actual per-hop bandwidth.
    """
    zero_ddl = 1e-5
    num_flow, num_link = route.shape
    actual_bandwidth = np.zeros((num_link,))
    zs_mask = shaping_delay < zero_ddl
    short_rate = np.where(zs_mask, flow_profile[:, 0], flow_profile[:, 1] / shaping_delay)
    burst = np.where(zs_mask, flow_profile[:, 1], 0)
    rate = np.concatenate((short_rate, flow_profile[:, 0] - short_rate))
    for link_idx in range(num_link):
        min_bd, link_rate = bandwidth_two_slope_(route[:, link_idx], ddl[:, link_idx], rate, burst, shaping_delay)
        actual_bandwidth[link_idx] = max(np.nanmax(min_bd), link_rate)
    return actual_bandwidth


def bandwidth_two_slope_(link_mask, link_ddl, rate, burst, shaping_delay):
    """
    Calculate the actual bandwidth (with two-slope shapers) at one hop.
    :param link_mask: mask to retrieve the subset of flows at this hop.
    :param link_ddl: the local deadlines of the flows at this hop.
    :param rate: the short-term and long-term rates of the flows.
    :param burst: the burst sizes of the flows.
    :param shaping_delay: the shaping delays of the flows.
    :return: the bandwidth requirement at the inflection points of the aggregate service curve,
             and the aggregate long-term rate.
    """
    zero_ddl = 1e-5
    num_flow = len(link_mask)
    link_ddl = np.concatenate((link_ddl, link_ddl + shaping_delay))
    link_mask = np.concatenate((link_mask, link_mask))
    rate_mask = np.concatenate((np.zeros((num_flow,), dtype=bool), np.ones((num_flow,), dtype=bool)))
    link_burst = np.concatenate((np.zeros((num_flow,)), burst))
    link_sort = np.argsort(link_ddl)
    link_mask = link_mask[link_sort]
    link_sort = link_sort[link_mask]
    link_ddl, link_rate = link_ddl[link_sort], rate[link_sort]
    rate_mask, link_burst = rate_mask[link_sort], link_burst[link_sort]
    rate_cum = np.cumsum(np.append(0, link_rate)[:-1])
    link_ddl_ = np.append(0, link_ddl)
    ddl_int = link_ddl_[1:] - link_ddl_[:-1]
    min_bd = rate_cum * ddl_int + link_burst
    min_bd = np.cumsum(min_bd)
    min_bd, link_ddl = min_bd[rate_mask], link_ddl[rate_mask]
    zero_mask = np.logical_and(min_bd < zero_ddl, link_ddl < zero_ddl)
    min_bd = min_bd / link_ddl
    min_bd = np.where(zero_mask, 0, min_bd)
    return min_bd, np.sum(link_rate)


def improve_one_slope(route, flow_profile, shaping_delay, ddl):
    """
    Improve the solution (with one-slope shapers) through reallocation of shaping delay and local deadlines.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param shaping_delay: the shaping delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the shaping delay and local deadlines after improvement.
    """
    # TODO: Implement the one-slope version.
    return shaping_delay, ddl, bandwidth_one_slope(route, flow_profile, shaping_delay, ddl)


def improve_two_slope(route, flow_profile, shaping_delay, ddl):
    """
    Improve the solution (with two-slope shapers) through reallocation of shaping delay and local deadlines.
    :param route: the network routes.
    :param flow_profile: the flow profile.
    :param shaping_delay: the shaping delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the shaping delay and local deadlines after improvement.
    """
    zero_ddl = 1e-5
    num_flow, num_link = route.shape
    long_rate, burst = flow_profile[:, 0], flow_profile[:, 1]
    actual_bandwidth = np.zeros((num_link,))
    for link_idx in range(num_link):
        # Retrieve the link related data.
        link_mask, link_ddl = route[:, link_idx], ddl[:, link_idx]
        link_sort = np.argsort(link_ddl)
        link_mask = link_mask[link_sort]
        link_sort = link_sort[link_mask]
        link_ddl, link_shaping = link_ddl[link_sort], shaping_delay[link_sort]
        link_long_rate, link_burst = long_rate[link_sort], burst[link_sort]
        # Compute the link bandwidth and room for reshaping for each flow.
        zs_mask = shaping_delay < zero_ddl
        short_rate = np.where(zs_mask, flow_profile[:, 0], flow_profile[:, 1] / shaping_delay)
        burst_mask = np.where(zs_mask, burst, 0)
        rate = np.concatenate((short_rate, flow_profile[:, 0] - short_rate))
        min_bd, link_rate = bandwidth_two_slope_(route[:, link_idx], ddl[:, link_idx], rate, burst_mask, shaping_delay)
        bandwidth = max(np.nanmax(min_bd), link_rate)
        actual_bandwidth[link_idx] = bandwidth
        link_sddl = np.sort(link_ddl + link_shaping)
        shaping_room = (bandwidth - min_bd) * link_sddl
        # Reshape each flow and update the shaping delay and local deadline.
        ddl1 = 0
        for flow_idx, (ddl2, s, r, b) in enumerate(zip(link_ddl, link_shaping, link_long_rate, link_burst)):
            flow_ddl = max(ddl1, ddl2 + s - b / r)
            left = bisect.bisect_right(link_sddl, flow_ddl + zero_ddl)
            right = bisect.bisect_left(link_sddl, ddl2 + s - zero_ddl)
            for x, y in zip(link_sddl[left:right], shaping_room[left:right]):
                y_max = max(0, x - ddl2) * b / s + y
                if y_max < b:
                    flow_ddl = max(flow_ddl, (b * x - y_max * (ddl2 + s)) / (b - y_max))
            # Update the data according to the reshaping decision.
            assert flow_ddl < ddl2 + zero_ddl
            link_ddl[flow_idx] = ddl1 = flow_ddl
            link_shaping[flow_idx] = ddl2 + s - flow_ddl
            left = bisect.bisect_right(link_sddl, flow_ddl + zero_ddl)
            for idx, x in enumerate(link_sddl[left:right]):
                shaping_room[left + idx] -= b * (max(0, x - flow_ddl) / (ddl2 + s - flow_ddl) - max(0, x - ddl2) / s)
        ddl[:, link_idx][link_sort] = link_ddl
        shaping_delay[link_sort] = link_shaping
    return shaping_delay, ddl, actual_bandwidth
