import numpy as np
from bisect import bisect_left
from scipy.optimize import root_scalar
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from lib.network_parser import get_objective, parse_solution

"""
Functions related to traffic reprofiling heuristics for network with static priority schedulers.
"""

NUM_CLASS = 8  # the maximum number of priority classes.


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
    num_flow, num_link = path_matrix.shape
    # Compute the maximum reprofiling delay.
    reprofiling_delay = np.concatenate(
        (flow_profile[:, 2][np.newaxis, :], flow_profile[:, 1][np.newaxis, :] / flow_profile[:, 0][np.newaxis, :]),
        axis=0)
    reprofiling_delay = np.amin(reprofiling_delay, axis=0)
    # Evenly split the remaining deadline onto each hop.
    ddl = ((flow_profile[:, 2] - reprofiling_delay) / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(path_matrix, ddl, 0)
    # Determine the priority class assignment at each hop.
    priority = np.zeros_like(ddl, dtype=int)
    for link_idx in range(num_link):
        link_mask = path_matrix[:, link_idx]
        link_priority = priority_assignment(ddl[:, link_idx][link_mask])
        priority[:, link_idx][link_mask] = link_priority
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl, priority)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth, bandwidth, priority


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
    # Evenly split the deadline onto each hop.
    ddl = (flow_profile[:, 2] / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl = np.where(path_matrix, ddl, 0)
    # Determine the priority class assignment at each hop.
    priority = np.zeros_like(ddl, dtype=int)
    for link_idx in range(num_link):
        link_mask = path_matrix[:, link_idx]
        link_priority = priority_assignment(ddl[:, link_idx][link_mask])
        priority[:, link_idx][link_mask] = link_priority
    bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl, priority)
    total_bandwidth = get_objective(bandwidth, objective, weight)
    return total_bandwidth, bandwidth, ddl, priority


def priority_assignment(ddl):
    """
    Assign flows to priority classes according to their local deadlines.
    :param ddl: the link delays (local deadlines) of the flows at one hop.
    :return: the priority class assignment.
    """
    ddl = ddl.reshape(-1, 1)
    # clustering_labels = []
    # inertias = []
    # silhouette_scores = []
    # Assign each flow to a different class if the number of flows is smaller than NUM_CLASS.
    if len(ddl) <= NUM_CLASS:
        centers = ddl.flatten()
        labels = np.arange(len(centers))
    else:
        kmeans = KMeans(n_clusters=NUM_CLASS, random_state=42, n_init='auto')
        kmeans.fit(ddl)
        # Ensure each data point is assigned to the nearest centroid.
        distances_to_centers = kmeans.transform(ddl)
        closest_center_indices = np.argmin(distances_to_centers, axis=1)
        # Reorder the cluster labels to ensure flows with smaller local deadlines get assigned to higher priority.
        centers = kmeans.cluster_centers_.flatten()
        # labels = kmeans.labels_
        labels = closest_center_indices
    # Sort centers to determine the correct order.
    sorted_indices = np.argsort(centers)
    # Merge cluster centroids that are close enough.
    center_map = {}
    prev_center, prev_idx = -float('inf'), -1
    for center_idx in sorted_indices:
        cur_center = centers[center_idx]
        if not np.isclose(cur_center, prev_center):
            prev_center, prev_idx = cur_center, center_idx
        center_map[center_idx] = prev_idx
    center_relabeled = np.array([center_map[label] for label in labels])
    # Build mapping old_label -> new_label and remove empty clusters.
    label_map, num_empty = {}, 0
    for new_label, old_label in enumerate(sorted_indices):
        if np.sum(center_relabeled == old_label) == 0:
            num_empty += 1
        else:
            label_map[old_label] = new_label + 1 - num_empty
    # Apply mapping to all labels.
    order_relabeled = np.array([label_map[label] for label in center_relabeled])
    # # Use the sum of squared distances to nearest cluster center to determine the best k.
    # inertias.append(kmeans.inertia_)
    # # Silhouette score method for choosing the best k.
    # # Silhouette score only defined for k > 1
    # if k > 1:
    #     score = silhouette_score(ddl, kmeans.labels_)
    #     silhouette_scores.append(score)
    # else:
    #     silhouette_scores.append(np.nan)
    # heuristic elbow detection.
    # best_k_elbow = np.argmin(np.gradient(np.gradient(inertias)))
    # best_k_silhouette = np.nanargmax(silhouette_scores)

    ################################### Test ###################################
    # TODO: to be removed later.
    prev_max = -float("inf")
    for class_idx in range(len(centers) - num_empty):
        class_min = np.amin(ddl[order_relabeled == class_idx + 1])
        if class_min < prev_max:
            assert class_min >= prev_max, "priority assignment test failed"
        prev_max = np.amax(ddl[order_relabeled == class_idx + 1])
    return order_relabeled


def bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl, priority):
    """
    Calculate the actual bandwidth according to the solution variables.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the link delays of the solution.
    :param priority: the priority class assignment.
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
        link_priority = priority[:, link_idx]
        link_mask = path_matrix[:, link_idx]
        link_bd = 0
        for priority_idx in range(1, np.amax(link_priority) + 1):
            link_priority_mask = np.zeros_like(link_mask, dtype=int)
            link_priority_mask[np.logical_and(link_mask, link_priority < priority_idx)] = 1
            link_priority_mask[link_priority == priority_idx] = 2
            priority_ddl = np.amin(ddl[:, link_idx][link_priority == priority_idx])
            min_bd, _, _, _, _ = bandwidth_two_slope_(link_priority_mask, priority_ddl, rate, burst, reprofiling_delay)
            link_bd = max(link_bd, min_bd)
        actual_bandwidth[link_idx] = link_bd
    return actual_bandwidth


def bandwidth_two_slope_(link_priority_mask, priority_ddl, rate, burst, reprofiling_delay):
    """
    Calculate the actual bandwidth at one hop.
    :param link_priority_mask: mask to retrieve the subset of flows from both the specified priority class and the higher
    priority class at this hop.
    :param priority_ddl: the local deadline of the specified priority class at this hop.
    :param rate: the short-term and long-term rates of the flows.
    :param burst: the burst sizes of the flows.
    :param reprofiling_delay: the reprofiling delays of the flows.
    :return: the function values at the inflection points of the aggregate service curve,
             and the aggregate long-term rate.
    """
    tor = 1e-15
    num_flow = len(link_priority_mask)
    # Retrieve the subset of flows from the specified priority class and all higher priority classes.
    hp_mask = link_priority_mask == 1
    priority_mask = link_priority_mask == 2
    reprofiling_sort = np.argsort(reprofiling_delay)
    # Split the higher priority classes into two groups based on the local deadline of the priority class.
    hp_mask1 = np.logical_and(hp_mask, reprofiling_delay < priority_ddl)
    hp_mask2 = np.logical_and(hp_mask, reprofiling_delay >= priority_ddl)
    hp_x, hp_y = np.zeros((np.sum(hp_mask),), dtype=float), np.zeros((np.sum(hp_mask),), dtype=float)
    # Compute the function values at the inflection points lower than the priority deadline.
    hp_short_rate = np.sum(rate[:num_flow][hp_mask])
    hp_mask1 = hp_mask1[reprofiling_sort]
    hp_sort = reprofiling_sort[hp_mask1]
    hp_rp, hp_rate, hp_burst = reprofiling_delay[hp_sort], rate[num_flow:][hp_sort], burst[hp_sort]
    hp_rate_cum = np.cumsum(np.append(hp_short_rate, hp_rate))
    hp_rp_ = np.append(0, hp_rp)
    hp_rp_ = np.append(hp_rp_, priority_ddl)
    hp_rp_int = hp_rp_[1:] - hp_rp_[:-1]
    hp_y1 = hp_rate_cum * hp_rp_int
    hp_y1[:-1] += hp_burst
    hp_y1 = np.cumsum(hp_y1)
    hp_x[:len(hp_rp)] = hp_rp
    hp_y[:len(hp_rp)] = hp_y1[:-1]
    # Compute the function values at the inflection points grater than the priority deadline.
    priority_short_rate = np.sum(rate[:num_flow][priority_mask]) + hp_rate_cum[-1]
    priority_hp2_mask = np.logical_or(priority_mask, hp_mask2)
    priority_delay = reprofiling_delay.copy()
    priority_delay[hp_mask2] -= priority_ddl
    priority_sort = np.argsort(priority_delay)
    priority_hp2_mask = priority_hp2_mask[priority_sort]
    priority_sort = priority_sort[priority_hp2_mask]
    php_mask = priority_mask[priority_sort]
    priority_rp = priority_delay[priority_sort]
    priority_rate, priority_burst = rate[num_flow:][priority_sort], burst[priority_sort]
    priority_rate_cum = np.cumsum(np.append(priority_short_rate, priority_rate)[:-1])
    priority_rp_ = np.append(0, priority_rp)
    priority_rp_int = priority_rp_[1:] - priority_rp_[:-1]
    priority_y = priority_rate_cum * priority_rp_int + priority_burst
    priority_y = np.cumsum(priority_y) + hp_y1[-1]
    hp_x[len(hp_rp):] = priority_rp[np.logical_not(php_mask)] + priority_ddl
    hp_y[len(hp_rp):] = priority_y[np.logical_not(php_mask)]
    priority_x = priority_rp[php_mask] + priority_ddl
    priority_y = priority_y[php_mask]
    # Compute the minimum bandwidth required.
    hp_x_mask = hp_x >= priority_ddl
    hp_zero_mask = np.logical_and(hp_x < tor, hp_y < tor)
    hp_x_mask = np.logical_and(hp_x_mask, np.logical_not(hp_zero_mask))
    hp_bd = np.divide(hp_y, hp_x, out=np.zeros_like(hp_x), where=hp_x_mask)
    priority_zero_mask = np.logical_and(priority_x < tor, priority_y < tor)
    priority_bd = np.divide(priority_y, priority_x, out=np.zeros_like(priority_x),
                            where=np.logical_not(priority_zero_mask))
    hp_burst_bd = 0 if hp_y1[-1] < tor and priority_ddl < tor else hp_y1[-1] / priority_ddl
    hp_bd = 0 if len(hp_bd) == 0 else np.nanmax(hp_bd)
    priority_bd = 0 if len(priority_bd) == 0 else np.nanmax(priority_bd)
    agg_rate = priority_short_rate + np.sum(priority_rate)
    bandwidth = max(hp_bd, priority_bd, hp_burst_bd, agg_rate)

    # Function to compute the intersection point between bandwidth and higher priority traffic (used as a lower bound
    # for reprofiling adjustment).
    def get_ddl_lb(link_bd):
        hp_x1 = np.append(hp_rp, priority_ddl)
        hp_y1_diff = link_bd * hp_x1 - hp_y1
        # For numerical stability.
        if -tor < hp_y1_diff[-1] < 0:
            hp_y1_diff[-1] = 0
        ################################### Test ###################################
        # TODO: to be removed later.
        if link_bd < bandwidth:
            assert False, "link bandwidth value test failed"
        if hp_y1_diff[-1] < 0:
            assert False, "bandwidth test failed"
        if hp_y1_diff[0] > 0:
            priority_ddl_lb = 0
        elif hp_y1_diff[0] == 0:
            priority_ddl_lb = hp_x1[0]
        else:
            if hp_y1_diff[-1] == 0:
                priority_ddl_lb = priority_ddl
            else:
                lb_idx = bisect_left(hp_y1_diff, 0) - 1
                priority_ddl_lb = hp_y1_diff[lb_idx] / (hp_rate_cum[lb_idx + 1] - link_bd) + hp_x1[lb_idx]
        return priority_ddl_lb

    return bandwidth, (hp_x, hp_y), (priority_x, priority_y), agg_rate, get_ddl_lb


def improve_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl):
    """
    Apply greedy reprofiling to improve the solution through reallocation of reprofiling delay and local deadlines.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param reprofiling_delay: the reprofiling delays of the solution.
    :param ddl: the local deadlines of the solution.
    :return: the reprofiling delay and local deadlines after improvement.
    """
    zero_tor = 1e-15
    num_flow, num_link = path_matrix.shape
    long_rate, burst = flow_profile[:, 0], flow_profile[:, 1]
    # Initialize link delays and priority assignment.
    ddl_new = np.zeros_like(ddl)
    actual_bandwidth = np.zeros((num_link,))
    priority_class = np.zeros_like(ddl, dtype=int)
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
        link_budget = ddl[:, link_idx] + reprofiling_delay
        # Determine the priority class assignment at this link.
        link_priority = priority_assignment(ddl[:, link_idx][link_mask])
        priority_class[:, link_idx][link_mask] = link_priority
        link_priority = priority_class[:, link_idx]
        # Perform adjustment for each priority class (starting from the highest priority).
        link_bd = 0
        for priority_idx in range(1, np.amax(link_priority) + 1):
            link_priority_mask = np.zeros_like(link_mask, dtype=int)
            link_priority_mask[np.logical_and(link_mask, link_priority < priority_idx)] = 1
            link_priority_mask[link_priority == priority_idx] = 2
            # Set the worst case delay of the priority class to the smallest local deadline.
            priority_ddl = np.amin(ddl[:, link_idx][link_priority == priority_idx])
            priority_reprofiling = np.minimum(link_budget[link_priority == priority_idx] - priority_ddl,
                                              burst[link_priority == priority_idx] / long_rate[
                                                  link_priority == priority_idx])
            reprofiling_delay[link_priority == priority_idx] = priority_reprofiling
            # Compute the short-term rate and burst size of flows (for bandwidth computation).
            zs_mask = reprofiling_delay < zero_tor
            reprofiling_delay[zs_mask] = 0
            short_rate = np.divide(burst, reprofiling_delay, out=np.copy(long_rate), where=np.logical_not(zs_mask))
            burst_mask = np.where(zs_mask, burst, 0)
            rate = np.concatenate((short_rate, long_rate - short_rate))
            min_bd, hp_xy, priority_xy, priority_rate, ddl_lb_getter = bandwidth_two_slope_(link_priority_mask,
                                                                                            priority_ddl, rate,
                                                                                            burst_mask,
                                                                                            reprofiling_delay)
            link_bd = max(link_bd, min_bd)
            priority_ddl_lb = ddl_lb_getter(link_bd)
            # Compute reprofiling room for each flow and perform adjustment.
            # Step 1: Collect the infection points of the flows that can impact the reprofiling room of the others
            # (influencers), and the inflection points whose reprofiling room get impacted (influencees).
            hp_mask = hp_xy[0] >= priority_ddl
            influencers = priority_xy[0]
            influencers_sort = np.argsort(reprofiling_delay)
            influencers_sort = influencers_sort[(link_priority == priority_idx)[influencers_sort]]
            influencees1 = hp_xy[0][np.logical_not(hp_mask)]
            influencees1_y = hp_xy[1][np.logical_not(hp_mask)]
            influencees2 = np.concatenate((priority_xy[0], hp_xy[0][hp_mask]))
            influencees2_y = np.concatenate((priority_xy[1], hp_xy[1][hp_mask]))
            influencees2_sort = np.argsort(influencees2)
            influencees = np.concatenate((influencees1, influencees2[influencees2_sort]))
            influence_map = np.argsort(influencees2_sort)[:len(priority_xy[0])] + len(influencees1)
            influencees_y = np.concatenate((influencees1_y, influencees2_y[influencees2_sort]))
            reprofiling_room = link_bd * influencees - influencees_y
            ################################### Test ###################################
            # TODO: to be removed later.
            if np.any(reprofiling_room[len(influencees1):] < -zero_tor):
                assert False, "reprofiling room test failed"
            # Step 2: Compute the initial states among the influencers and influencees (which one gets influenced by
            # which).
            influence_matrix = np.zeros((len(influencers), len(influencees)), dtype=float)
            checkpoints = []
            infuencer_burst, influencer_rate = burst[influencers_sort], long_rate[influencers_sort]
            influencer_reprofiling = infuencer_burst / influencer_rate
            influencer_fr_points = priority_xy[0] - influencer_reprofiling
            influencer_fr_mask = influencer_fr_points >= priority_ddl
            for influencer_idx, fr_point in enumerate(influencer_fr_points):
                if not influencer_fr_mask[influencer_idx]:
                    if fr_point > priority_ddl_lb:
                        checkpoints.append((fr_point, 0, influencer_idx))
            for influencee_idx, influencee in enumerate(influencees1):
                if influencee > priority_ddl_lb:
                    checkpoints.append((influencee, 1, influencee_idx))
            checkpoints.append((priority_ddl_lb,))
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            # Step 3: Loop through each checkpoint and see if reprofiling delay can be adjusted.
            prev_ddl = priority_ddl
            # Set small negative values to zero for numerical stability.
            prev_room = np.where(np.logical_and(reprofiling_room < zero_tor, reprofiling_room > -zero_tor), 0,
                                 reprofiling_room)
            num_not_influenced = len(influencees1)
            for checkpoint in checkpoints:
                cur_ddl = checkpoint[0]
                delta = prev_ddl - cur_ddl
                if delta > 0:
                    # Parameters for back-tracking functions.
                    func_params = [[0, [], [], 0] for _ in range(len(influencees))]
                    # Compute the amount of increment that each influencer adds to each influencee.
                    for influencer_idx in range(len(influencers)):
                        if influencer_fr_mask[influencer_idx]:
                            increment = delta * influencer_rate[influencer_idx]
                            influence_matrix[influencer_idx, num_not_influenced:] = increment
                            # Update the function parameters.
                            for i in range(num_not_influenced, len(influencees)):
                                func_params[i][0] += increment
                        else:
                            influencee_idx = influence_map[influencer_idx]
                            dist_prev = influencees[influencee_idx] - prev_ddl
                            dist_cur = influencees[influencee_idx] - cur_ddl
                            full_increment = delta * influencees_y[influencee_idx] / dist_cur
                            increment_degree = np.zeros((influencee_idx - num_not_influenced,), dtype=float)
                            if dist_prev > 0:
                                increment_degree = (influencees[influencee_idx] - influencees[
                                                                                  num_not_influenced:influencee_idx]) / dist_prev
                            increment = increment_degree * full_increment
                            influence_matrix[influencer_idx, num_not_influenced:influencee_idx] = increment
                            ################################### Test ###################################
                            # TODO: to be removed later.
                            if not np.all(increment >= 0):
                                assert False, "increment computation test failed"
                            # Update the function parameters.
                            for i, inc in enumerate(increment, num_not_influenced):
                                if dist_prev > 0:
                                    func_params[i][1].append(inc * dist_prev * dist_cur / (delta ** 2))
                                    func_params[i][2].append(dist_prev / delta)
                                func_params[i][3] += inc * dist_cur / delta

                                ################################### Test ###################################
                                # TODO: to be removed later.
                                def test_func(x):
                                    if np.isclose(inc, 0):
                                        return 0
                                    b = inc * dist_prev * dist_cur / (delta ** 2)
                                    c = dist_prev / delta
                                    d = inc * dist_cur / delta
                                    if dist_prev == 0 and x == 0:
                                        return d
                                    return d - b / (x + c)

                                if not (np.isclose(test_func(0), 0) and np.isclose(test_func(1), inc)):
                                    assert False, "test_func failed"

                    # Check if the reprofiling room of any inflection point is depleted.
                    cur_room = prev_room - np.sum(influence_matrix, axis=0)
                    # Set small negative values to zero for numerical stability.
                    cur_room = np.where(np.logical_and(cur_room < zero_tor, cur_room > -zero_tor), 0, cur_room)
                    if np.any(cur_room[num_not_influenced:] < 0):
                        # Step 4: Track back to identify the tightest possible deadline for the priority class.
                        min_x = float("inf")
                        for influencee_idx in np.arange(num_not_influenced, len(influencees))[
                            cur_room[num_not_influenced:] < 0]:
                            if prev_room[influencee_idx] == 0:
                                min_x = 0
                                break
                            a, b, c, d = func_params[influencee_idx]
                            b, c = np.asarray(b, dtype=float), np.asarray(c, dtype=float)
                            d -= prev_room[influencee_idx]

                            def influencee_func(x):
                                return a * x - np.sum(b / (x + c)) + d

                            ################################### Test ###################################
                            # TODO: to be removed later.
                            if not (np.isclose(influencee_func(0), -prev_room[influencee_idx]) and np.isclose(
                                    influencee_func(1), -cur_room[influencee_idx])):
                                assert False, "influencee_func test failed"
                            if influencee_func(0) * influencee_func(1) > 0:
                                assert False, "influencee boundary test failed"
                            sol = root_scalar(influencee_func, bracket=(0, 1), method='brentq')
                            ################################### Test ###################################
                            # TODO: to be removed later.
                            if not sol.converged:
                                assert False, "convergence test failed"
                            min_x = min(min_x, sol.root)
                        priority_ddl = prev_ddl - min_x * delta
                        break
                    prev_room = cur_room
                # Update the state of the influencers and the influencees.
                if len(checkpoint) > 1:
                    if checkpoint[1] == 0:
                        influencer_fr_mask[checkpoint[2]] = True
                    else:
                        num_not_influenced -= 1
                priority_ddl, prev_ddl = cur_ddl, cur_ddl
            # Update the link delays and reprofiling delays.
            ddl_new[link_priority == priority_idx, link_idx] = priority_ddl
            priority_reprofiling = np.minimum(link_budget[link_priority == priority_idx] - priority_ddl,
                                              burst[link_priority == priority_idx] / long_rate[
                                                  link_priority == priority_idx])
            reprofiling_delay[link_priority == priority_idx] = priority_reprofiling
            ################################### Test ###################################
            # TODO: to be removed later.
            zs_mask = reprofiling_delay < zero_tor
            reprofiling_delay[zs_mask] = 0
            short_rate = np.divide(burst, reprofiling_delay, out=np.copy(long_rate), where=np.logical_not(zs_mask))
            burst_mask = np.where(zs_mask, burst, 0)
            rate = np.concatenate((short_rate, long_rate - short_rate))
            min_bd_new, hp_xy_new, priority_xy_new, _, _ = bandwidth_two_slope_(link_priority_mask, priority_ddl, rate,
                                                                                burst_mask, reprofiling_delay)
            if not (min_bd_new < link_bd or np.isclose(min_bd_new, link_bd, atol=1e-03)):
                # hp_mask_new = hp_xy_new[0] >= priority_ddl
                # influencees1_new = hp_xy_new[0][np.logical_not(hp_mask_new)]
                # influencees1_y_new = hp_xy_new[1][np.logical_not(hp_mask_new)]
                # influencees2_new = np.concatenate((priority_xy_new[0], hp_xy_new[0][hp_mask_new]))
                # influencees2_y_new = np.concatenate((priority_xy_new[1], hp_xy_new[1][hp_mask_new]))
                # influencees2_sort_new = np.argsort(influencees2_new)
                # influencees_new = np.concatenate((influencees1_new, influencees2_new[influencees2_sort_new]))
                # influencees_y_new = np.concatenate((influencees1_y_new, influencees2_y_new[influencees2_sort_new]))
                # reprofiling_room_new = link_bd * influencees_new - influencees_y_new
                # bandwidth_two_slope_(link_priority_mask, priority_ddl, rate, burst_mask, reprofiling_delay)
                assert False, "bandwidth test failed"
        actual_bandwidth[link_idx] = link_bd
    # # Compute the actual bandwidth after one iteration of traffic smoothing.
    # actual_bandwidth = bandwidth_two_slope(path_matrix, flow_profile, reprofiling_delay, ddl_new, priority_class)
    # Deadline reallocation.
    ddl_unused = flow_profile[:, 2] - np.sum(ddl_new, axis=1) - reprofiling_delay
    ddl_per_hop = (ddl_unused / np.sum(path_matrix, axis=1))[:, np.newaxis] * np.ones((num_link,))
    ddl_new += np.where(path_matrix, ddl_per_hop, 0)
    return reprofiling_delay, ddl_new, actual_bandwidth, priority_class
