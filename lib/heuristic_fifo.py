import numpy as np

from lib.network_parser import get_objective, parse_solution

"""
Functions related to traffic reprofiling heuristics for network with FIFO schedulers.
"""


def full_reprofiling(path_matrix, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by the full reprofiling (FR) solution.
    Equivalent to the rate proportional processor sharing (RPPS) solution with FIFO schedulers.
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
    return total_bandwidth, bandwidth


def no_reprofiling(path_matrix, flow_profile, objective, weight):
    """
    Compute the minimum bandwidth required by the no reprofiling (NR) solution.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the minimum bandwidth.
    """
    # TODO: Test with larger network topology.
    num_flow, num_link = path_matrix.shape
    reprofiling_delay = np.zeros((num_flow,))
    ddl = np.zeros((num_link,))
    ddl_mask = np.zeros((num_link,), dtype=bool)
    # Compute the local deadline at each hop by selecting the smallest local deadline among the flows in each iteration.
    for _ in range(num_flow):
        flow_ddl = np.ones((num_flow,)) * np.inf
        for flow_idx in range(num_flow):
            allocated_ddl = np.sum(ddl[path_matrix[flow_idx]])
            remaining_ddl = flow_profile[flow_idx, 2] - allocated_ddl
            vacant_link = np.logical_and(path_matrix[flow_idx], np.logical_not(ddl_mask))
            # Compute the local deadline by evenly allocating the remaining deadline of each flow.
            if np.sum(vacant_link) > 0:
                flow_ddl[flow_idx] = remaining_ddl / np.sum(vacant_link)
        # Select the smallest local deadline among all the flows.
        small_idx = np.argmin(flow_ddl)
        ddl[np.logical_and(path_matrix[small_idx], np.logical_not(ddl_mask))] = flow_ddl[small_idx]
        ddl_mask[path_matrix[small_idx]] = True
        # Finish the iteration if all the links get a local deadline assigned.
        if np.all(ddl_mask):
            break
    assert np.all(ddl_mask)
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth, bandwidth


def check_solution(path_matrix, flow_profile, solution):
    """
    Check whether the actual bandwidth is consistent with the solution result.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param solution: solution returned by the NLP solver.
    :return: consistency of the solution.
    """
    err = 1e-3
    num_flow, num_link = path_matrix.shape
    reprofiling_delay, ddl, bandwidth = parse_solution(path_matrix, solution)
    # Check if solution deadlines are non-negative.
    feasible1 = np.all(reprofiling_delay >= 0) and np.all(ddl >= 0)
    # Check if each flow in-network deadline stays in range.
    total_ddl, sd_ub = flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0]
    net_ddl = np.array([np.sum(ddl[path_matrix[flow_idx]]) for flow_idx in range(num_flow)])
    feasible2 = np.all(net_ddl + reprofiling_delay <= total_ddl + err)
    feasible3 = np.all(reprofiling_delay <= sd_ub + err)
    feasible = feasible1 and feasible2 and feasible3
    # Check if the computed bandwidth is consistent with the returned bandwidth.
    actual_bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    tight = np.all(np.abs(actual_bandwidth - bandwidth) <= err)
    return feasible and tight


def bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl):
    """
    Calculate the actual bandwidth according to the solution variables.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the link delays of the solution.
    :return: the actual per-hop bandwidth.
    """
    zero_ddl = 1e-15
    num_flow, num_link = path_matrix.shape
    actual_bandwidth = np.zeros((num_link,))
    zs_mask = reprofiling_delay < zero_ddl
    reprofiling_delay[zs_mask] = 0
    short_rate = np.divide(flow_profile[:, 1], reprofiling_delay, out=np.copy(flow_profile[:, 0]),
                           where=np.logical_not(zs_mask))
    burst = np.where(zs_mask, flow_profile[:, 1], 0)
    rate = np.concatenate((short_rate, flow_profile[:, 0] - short_rate))
    for link_idx in range(num_link):
        min_bd, link_rate = bandwidth_two_slope_(path_matrix[:, link_idx], ddl[link_idx], rate, burst,
                                                 reprofiling_delay)
        actual_bandwidth[link_idx] = max(np.nanmax(min_bd), link_rate)
    return actual_bandwidth


def bandwidth_two_slope_(link_mask, link_ddl, rate, burst, reprofiling_delay):
    """
    Calculate the actual bandwidth at one hop.
    :param link_mask: mask to retrieve the subset of flows at this hop.
    :param link_ddl: the local deadline of the flows at this hop.
    :param rate: the short-term and long-term rates of the flows.
    :param burst: the burst sizes of the flows.
    :param reprofiling_delay: the reprofiling delays of the flows.
    :return: the bandwidth requirement at the inflection points of the aggregate service curve,
             and the aggregate long-term rate.
    """
    zero_ddl = 1e-10
    num_flow = len(link_mask)
    short_rate = np.sum(rate[:num_flow][link_mask])
    link_sort = np.argsort(reprofiling_delay)
    link_mask = link_mask[link_sort]
    link_sort = link_sort[link_mask]
    link_rp, link_rate, link_burst = reprofiling_delay[link_sort], rate[num_flow:][link_sort], burst[link_sort]
    rate_cum = np.cumsum(np.append(short_rate, link_rate)[:-1])
    link_rp_ = np.append(0, link_rp)
    rp_int = link_rp_[1:] - link_rp_[:-1]
    min_bd = rate_cum * rp_int + link_burst
    min_bd = np.cumsum(min_bd)
    zero_mask = np.logical_and(min_bd < zero_ddl, link_rp + link_ddl < zero_ddl)
    min_bd = np.divide(min_bd, link_rp + link_ddl, out=np.zeros_like(min_bd), where=np.logical_not(zero_mask))
    return min_bd, short_rate + np.sum(link_rate)


def improve_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl):
    """
    Apply greedy reprofiling to improve the solution through reallocation of reprofiling delay and local deadlines.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the reprofiling delay and local deadlines after improvement.
    """
    zero_ddl = 1e-15
    num_flow, num_link = path_matrix.shape
    long_rate, burst = flow_profile[:, 0], flow_profile[:, 1]
    # Transform ddl into a set of local deadlines if ddl specifies a set of link deadlines.
    if ddl.ndim == 1:
        ddl = np.ones((num_flow, 1)) * ddl
        ddl = np.where(path_matrix, ddl, 0)
    # Initialize link delays.
    delay = np.zeros((num_link,))
    # Pre-processing to determine the order to improve the links.
    # TODO: try a different order to visit links.
    num_cover = np.zeros((num_link,), dtype=int)
    for link_idx in range(num_link):
        sub_net = path_matrix[path_matrix[:, link_idx]]
        num_cover[link_idx] = np.sum(np.any(sub_net, axis=0))
    link_order = np.arange(num_link)[np.argsort(-num_cover)]
    for link_idx in link_order:
        # Retrieve the link related data.
        link_mask = path_matrix[:, link_idx]
        link_ddl, link_reprofiling = ddl[:, link_idx][link_mask], reprofiling_delay[link_mask]
        link_long_rate, link_burst = long_rate[link_mask], burst[link_mask]
        link_budget = link_ddl + link_reprofiling
        # Set the worst case link delay to the smallest local deadline.
        link_delay = np.amin(link_ddl)
        link_reprofiling = np.minimum(link_budget - link_delay, link_burst / link_long_rate)
        reprofiling_delay[link_mask] = link_reprofiling
        # Compute the link bandwidth and room for reprofiling for each flow.
        zs_mask = reprofiling_delay < zero_ddl
        reprofiling_delay[zs_mask] = 0
        short_rate = np.divide(burst, reprofiling_delay, out=np.copy(long_rate), where=np.logical_not(zs_mask))
        burst_mask = np.where(zs_mask, burst, 0)
        rate = np.concatenate((short_rate, long_rate - short_rate))
        min_bd, link_rate = bandwidth_two_slope_(path_matrix[:, link_idx], link_delay, rate, burst_mask,
                                                 reprofiling_delay)
        bandwidth = max(np.nanmax(min_bd), link_rate)
        reprofiling_room = np.append(bandwidth - min_bd, bandwidth - link_rate)
        # Reprofile all the flows and update the reprofiling delay.
        room_mask = reprofiling_room < zero_ddl
        right_idx = np.arange(len(room_mask))[room_mask[::-1]][0]
        if right_idx == 0:
            # Set the link delay to 0 since all the flows can be reprofiled to their long-term rates.
            link_delay = 0
            link_reprofiling = link_burst / link_long_rate
        elif right_idx == 1:
            # The case when minimum bandwidth is achieved at the last bandwidth checkpoint.
            link_delay1, link_delay3 = 0, 0
            link_delay2 = np.amax(link_delay + link_reprofiling - link_burst / link_long_rate)
            if len(link_reprofiling) >= 2:
                link_sort = np.argsort(link_reprofiling)
                link_reprofiling_sort, link_burst_sort = link_reprofiling[link_sort], link_burst[link_sort]
                link_room = reprofiling_room[-3] * (link_reprofiling_sort[-2] + link_delay)
                if link_room < zero_ddl:
                    delta = 0
                else:
                    denominator = (link_burst_sort[-1] * (
                            link_reprofiling_sort[-1] - link_reprofiling_sort[-2]) - link_room *
                                   link_reprofiling_sort[-1])
                    delta = np.inf if denominator < zero_ddl else (link_room * link_reprofiling_sort[
                        -1] ** 2) / denominator
                link_delay3 = link_delay - delta
            link_delay = max(link_delay1, link_delay2, link_delay3)
            link_reprofiling = np.minimum(link_budget - link_delay, link_burst / link_long_rate)
        # Update the link delays and reprofiling delays.
        delay[link_idx] = link_delay
        reprofiling_delay[link_mask] = link_reprofiling
        # TODO: deadline reallocation.
    # Compute the actual bandwidth after one iteration of traffic smoothing.
    actual_bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, delay)
    return reprofiling_delay, delay, actual_bandwidth
