import numpy as np
import bisect

from lib.network_parser import get_objective

"""
Functions related to traffic reprofiling heuristics for network with SCED schedulers.
"""


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
        (flow_profile[:, 2][np.newaxis, :], flow_profile[:, 1][np.newaxis, :] / flow_profile[:, 0][np.newaxis, :]),
        axis=0)
    reprofiling_delay = np.amin(reprofiling_delay, axis=0)
    ddl = ((flow_profile[:, 2] - reprofiling_delay) / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(path_matrix, ddl, 0)
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth, bandwidth, None


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
    return total_bandwidth, bandwidth, ddl, None


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
    Apply greedy reprofiling to improve the solution through reallocation of reprofiling delay and local deadlines.
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
    return reprofiling_delay, ddl, actual_bandwidth, None
