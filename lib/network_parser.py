import numpy as np
import bisect
from concurrent.futures import ThreadPoolExecutor as TPE
from itertools import repeat


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


def rate_proportional(path_matrix, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by the rate proportional processor sharing (RPPS) solution.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the minimum bandwidth.
    """
    flow_rate = np.concatenate(
        (flow_profile[:, 0][np.newaxis, :], flow_profile[:, 1][np.newaxis, :] / flow_profile[:, 2][np.newaxis, :]),
        axis=0)
    flow_rate = np.amax(flow_rate, axis=0)
    bandwidth = np.sum(path_matrix * flow_rate[:, np.newaxis], axis=0)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth


def full_reprofiling(path_matrix, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by the full reprofiling (FR) solution.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the minimum bandwidth.
    """
    num_flow, num_link = path_matrix.shape
    reprofiling_delay = np.concatenate(
        (path_matrix[:, 2][np.newaxis, :], path_matrix[:, 1][np.newaxis, :] / path_matrix[:, 0][np.newaxis, :]),
        axis=0)
    reprofiling_delay = np.amin(reprofiling_delay, axis=0)
    ddl = ((flow_profile[:, 2] - reprofiling_delay) / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(path_matrix, ddl, 0)
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth


def no_reprofiling(path_matrix, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by the no reprofiling (NR) solution.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the minimum bandwidth.
    """
    num_flow, num_link = path_matrix.shape
    reprofiling_delay = np.zeros((num_flow,))
    ddl = (flow_profile[:, 2] / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(path_matrix, ddl, 0)
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth


def greedy(path_matrix, flow_profile, objective, weight, k=4, num_iter=2, num_workers=None):
    """
    Use importance sampling + greedy algorithm to efficiently approximate the optimal solution (guaranteed to be no
    worse than both the full reprofiling solution and the no reprofiling solution).
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :param k: number of (intermediate) initial solutions to sample (uniformly) from the importance region
        in each iteration.
    :param num_iter: number of iterations to refine the importance region.
    :param num_workers: number of workers for parallel computing.
    :return: the best solution found and its corresponding reprofiling ratio.
    """
    # Make sure that the importance sampling parameters are valid.
    assert num_iter >= 1, "Please set at least 1 iteration for the greedy search."
    assert k >= 0, "Please select a non-negative number of initial solutions to explore in greedy search."
    if num_iter > 1:
        assert k > 1, "Please select more than 1 initial solution if you want more than 1 iteration of greedy search."
    # Declare variables to keep the best solution.
    best_solution, best_reprofiling, best_ddl, best_ratio = np.inf, None, None, None
    # Specify the two extremes of the importance region.
    low_ratio, high_ratio = 0, 1
    for _ in range(num_iter):
        # Uniformly sample a range of reprofiling ratios between the two extremes.
        reprofiling_ratios = np.linspace(low_ratio, high_ratio, num=k + 2)
        # Apply the greedy algorithm to improve each initial solution in parallel.
        with TPE(max_workers=num_workers) as pool:
            solutions = list(
                pool.map(greedy_, repeat(path_matrix), repeat(flow_profile), reprofiling_ratios, repeat(objective),
                         repeat(weight)))
        # Retrieve the best solution in the current iteration.
        solution_sort = sorted(range(len(solutions)), key=lambda x: solutions[x][0])
        best_idx = solution_sort[0]
        # Update the best solution and the two extremes of the importance region.
        if solutions[best_idx][0] < best_solution:
            best_solution, best_reprofiling, best_ddl = solutions[best_idx]
            best_ratio = reprofiling_ratios[best_idx]
        low_ratio = reprofiling_ratios[max(0, best_idx - 1)]
        high_ratio = reprofiling_ratios[min(k + 1, best_idx + 1)]
    return best_solution, best_reprofiling, best_ddl, best_ratio


def greedy_(path_matrix, flow_profile, reprofiling_ratio, objective, weight):
    """
    Use the heuristic-based greedy algorithm to improve a given initial solution (specified by the reprofiling ratio).
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_ratio: how much reprofiling to apply to the initial solution. Should be a real number between 0
        and 1, with 0 for no reprofiling and 1 for full reprofiling.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the solution that Greedy converges to.
    """
    rate, burst, deadline = flow_profile[:, 0], flow_profile[:, 1], flow_profile[:, 2]
    # Compute the full reprofiling delay of each flow.
    num_flow, num_link = path_matrix.shape
    full_reprofiling_delay = np.concatenate((deadline[np.newaxis, :], burst[np.newaxis, :] / rate[np.newaxis, :]),
                                            axis=0)
    full_reprofiling_delay = np.amin(full_reprofiling_delay, axis=0)
    # Compute the reprofiling delay for the initial solution according to the reprofiling ratio.
    reprofiling_delay = full_reprofiling_delay * reprofiling_ratio
    # Evenly split the remaining network deadline to each hop.
    ddl = ((deadline - reprofiling_delay) / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(path_matrix, ddl, 0)
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    # Improve the deadline assignment using the greedy heuristic.
    new_solution, reprofiling_delay, ddl = improve_solution(path_matrix, flow_profile,
                                                            (reprofiling_delay, ddl, bandwidth), objective, weight)
    return new_solution, reprofiling_delay, ddl


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
    ddl = np.zeros_like(path_matrix, dtype=float)
    for flow_idx in range(num_flow):
        reprofiling_delay[flow_idx] = solution[f"D{flow_idx}"]
    for link_idx in range(num_link):
        bandwidth[link_idx] = solution[f"C{link_idx}"]
    for flow_idx, flow_route in enumerate(path_matrix):
        for link_idx, flow_link in enumerate(flow_route):
            if flow_link:
                ddl[flow_idx, link_idx] = solution[f"T{flow_idx}_{link_idx}"]
    return reprofiling_delay, ddl, bandwidth


"""
The following utility functions can be used for multiple supplementary purposes
and are not actively used in this implementation.
"""


def check_solution(path_matrix, flow_profile, solution):
    """
    Check whether the actual bandwidth is consistent with the solution result.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param solution: solution returned by the NLP solver.
    :return: consistency of the solution.
    """
    err = 1e-3
    reprofiling_delay, ddl, bandwidth = parse_solution(path_matrix, solution)
    # Check if solution deadlines are non-negative.
    feasible1 = np.all(reprofiling_delay >= 0) and np.all(ddl >= 0)
    # Check if each flow in-network deadline stays in range.
    total_ddl, sd_ub = flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0]
    net_ddl = np.sum(ddl, axis=1)
    feasible2 = np.all(net_ddl + reprofiling_delay <= total_ddl + err)
    feasible3 = np.all(reprofiling_delay <= sd_ub + err)
    feasible = feasible1 and feasible2 and feasible3
    # Check if the computed bandwidth is consistent with the returned bandwidth.
    actual_bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    tight = np.all(np.abs(actual_bandwidth - bandwidth) <= err)
    return feasible and tight


def improve_solution(path_matrix, flow_profile, solution, objective, weight):
    """
    Improve the solution using a heuristic-based iterative algorithm.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param solution: reprofiling delay, per-hop deadline, and bandwidth assigned by a solution.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the improved solution.
    """
    err = 1e-3
    reprofiling_delay, ddl, bandwidth = solution
    prev_bandwidth = np.inf
    # Run the greedy re(re)profiling algorithm iteratively until the solution cannot be further improved.
    while True:
        current_bandwidth = get_objective(bandwidth, objective, weight)
        # Break when the improvement is marginal.
        if prev_bandwidth - current_bandwidth < err:
            break
        prev_bandwidth = current_bandwidth
        # Improve the solution.
        reprofiling_delay, ddl, bandwidth = improve_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    return current_bandwidth, reprofiling_delay, ddl


def bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl):
    """
    Calculate the actual bandwidth according to the solution variables.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the actual per-hop bandwidth.
    """
    zero_ddl = 1e-5
    num_flow, num_link = path_matrix.shape
    actual_bandwidth = np.zeros((num_link,))
    zs_mask = reprofiling_delay < zero_ddl
    short_rate = np.divide(flow_profile[:, 1], reprofiling_delay, out=np.copy(flow_profile[:, 0]),
                           where=np.logical_not(zs_mask))
    burst = np.where(zs_mask, flow_profile[:, 1], 0)
    rate = np.concatenate((short_rate, flow_profile[:, 0] - short_rate))
    for link_idx in range(num_link):
        min_bd, link_rate = bandwidth_two_slope_(path_matrix[:, link_idx], ddl[:, link_idx], rate, burst,
                                                 reprofiling_delay)
        actual_bandwidth[link_idx] = max(np.nanmax(min_bd), link_rate)
    return actual_bandwidth


def bandwidth_two_slope_(link_mask, link_ddl, rate, burst, reprofiling_delay):
    """
    Calculate the actual bandwidth at one hop.
    :param link_mask: mask to retrieve the subset of flows at this hop.
    :param link_ddl: the local deadlines of the flows at this hop.
    :param rate: the short-term and long-term rates of the flows.
    :param burst: the burst sizes of the flows.
    :param reprofiling_delay: the reprofiling delays of the flows.
    :return: the bandwidth requirement at the inflection points of the aggregate service curve,
             and the aggregate long-term rate.
    """
    zero_ddl = 1e-5
    num_flow = len(link_mask)
    link_ddl = np.concatenate((link_ddl, link_ddl + reprofiling_delay))
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
    min_bd = np.divide(min_bd, link_ddl, out=np.zeros_like(min_bd), where=np.logical_not(zero_mask))
    return min_bd, np.sum(link_rate)


def improve_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl):
    """
    Apply greedy re(re)profiling to improve the solution through reallocation of reprofiling delay and local deadlines.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the reprofiling delay and local deadlines after improvement.
    """
    zero_ddl = 1e-6
    num_flow, num_link = path_matrix.shape
    long_rate, burst = flow_profile[:, 0], flow_profile[:, 1]
    # Pre-processing to determine the order to improve the links.
    num_cover = np.zeros((num_link,), dtype=int)
    for link_idx in range(num_link):
        sub_net = path_matrix[path_matrix[:, link_idx]]
        num_cover[link_idx] = np.sum(np.any(sub_net, axis=0))
    link_order = np.arange(num_link)[np.argsort(-num_cover)]
    for link_idx in link_order:
        # Retrieve the link related data.
        link_mask, link_ddl = path_matrix[:, link_idx], ddl[:, link_idx]
        link_sort = np.argsort(-(link_ddl + reprofiling_delay))
        link_mask = link_mask[link_sort]
        link_sort = link_sort[link_mask]
        link_ddl, link_reprofiling = link_ddl[link_sort], reprofiling_delay[link_sort]
        link_long_rate, link_burst = long_rate[link_sort], burst[link_sort]
        # Compute the link bandwidth and room for reprofiling for each flow.
        zs_mask = reprofiling_delay < zero_ddl
        short_rate = np.divide(flow_profile[:, 1], reprofiling_delay, out=np.copy(flow_profile[:, 0]),
                               where=np.logical_not(zs_mask))
        burst_mask = np.where(zs_mask, burst, 0)
        rate = np.concatenate((short_rate, flow_profile[:, 0] - short_rate))
        min_bd, link_rate = bandwidth_two_slope_(path_matrix[:, link_idx], ddl[:, link_idx], rate, burst_mask,
                                                 reprofiling_delay)
        bandwidth = max(np.nanmax(min_bd), link_rate)
        link_sddl = np.sort(link_ddl + link_reprofiling)
        reprofiling_room = (bandwidth - min_bd) * link_sddl
        # Reprofile each flow and update the reprofiling delay and local deadline.
        for flow_idx, (ddl2, s, r, b) in enumerate(zip(link_ddl, link_reprofiling, link_long_rate, link_burst)):
            flow_ddl = max(0, ddl2 + s - b / r)
            left = bisect.bisect_right(link_sddl, flow_ddl + zero_ddl)
            right = bisect.bisect_left(link_sddl, ddl2 + s - zero_ddl)
            for x, y in zip(link_sddl[left:right], reprofiling_room[left:right]):
                y_max = y if s == 0 else max(0, x - ddl2) * b / s + y
                if y_max < b:
                    flow_ddl = max(flow_ddl, (b * x - y_max * (ddl2 + s)) / (b - y_max))
            # Update the data according to the reprofiling decision.
            assert flow_ddl < ddl2 + zero_ddl
            link_ddl[flow_idx] = flow_ddl
            link_reprofiling[flow_idx] = ddl2 + s - flow_ddl
            left = bisect.bisect_right(link_sddl, flow_ddl + zero_ddl)
            for idx, x in enumerate(link_sddl[left:right]):
                x_old = 0 if s == 0 else max(0, x - ddl2) / s
                reprofiling_room[left + idx] -= b * ((x - flow_ddl) / (ddl2 + s - flow_ddl) - x_old)
        ddl[:, link_idx][link_sort] = link_ddl
        reprofiling_delay[link_sort] = link_reprofiling
    # Compute the actual bandwidth after one iteration of traffic smoothing.
    actual_bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    return reprofiling_delay, ddl, actual_bandwidth
