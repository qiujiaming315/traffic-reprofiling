from collections import defaultdict
import numpy as np
import os
from pathlib import Path

"""Utility functions."""


def load_net(route_path, flow_path, aggregate=False, aggregate_path=""):
    """
    Load data and do sanity check.
    :param route_path: path to the flow routes.
    :param flow_path: path to the flow profile.
    :param aggregate: whether the flows with same route and deadline class should be aggregated.
    :param aggregate_path: path to save the aggregated flow routes and profiles.
    :return: the loaded data.
    """
    route_data, flow_data = np.load(route_path), np.load(flow_path)
    flow_routes, flow_routes_pruned = route_data, None
    flow_profile, per_hop = flow_data['flow'], flow_data['per_hop']
    if type(route_data) is np.lib.npyio.NpzFile:
        # Parse the flow route data.
        flow_routes = route_data["routes"]
        if "routes_pruned" in route_data.files:
            # Load the pruned flow routes.
            flow_routes_pruned = route_data["routes_pruned"]
        if "app_dest_num" in route_data.files:
            # Multiply each flow according to the number of destination node(s) it has.
            flow_profile_expanded = np.zeros((0, flow_profile.shape[1]))
            for flow, dn in zip(flow_profile, route_data["app_dest_num"]):
                flow_profile_expanded = np.concatenate(
                    (flow_profile_expanded, np.ones((dn, flow_profile.shape[1])) * flow), axis=0)
            flow_profile = flow_profile_expanded
    if aggregate:
        flow_routes, flow_profile_agg = aggregate_flow(flow_routes, flow_profile)
        flow_routes_pruned, _ = aggregate_flow(flow_routes_pruned, flow_profile)
        flow_profile = flow_profile_agg
        if aggregate_path is not "":
            # Save the aggregated flow route and profile.
            Path(aggregate_path).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(aggregate_path, "flow_route.npy"), flow_routes)
            np.save(os.path.join(aggregate_path, "flow_route_pruned.npy"), flow_routes_pruned)
            np.savez(os.path.join(aggregate_path, "flow_profile.npy"), flow=flow_profile, per_hop=per_hop)
    net_type = flow_routes.dtype
    assert net_type is np.dtype(bool) or np.issubdtype(net_type, np.integer), f"Incorrect data type ({net_type}) " + \
                                                                              "for flow routes. Expect bool or int."
    assert flow_routes.ndim == 2, f"Incorrect dimension number ({flow_routes.ndim}) for flow routes. Expect 2."
    assert np.all(np.sum(flow_routes > 0,
                         axis=1) >= 2), "Each flow route should pass through at least 2 nodes."
    if net_type is np.dtype(int):
        assert np.all(flow_routes >= 0), "For flow routes from cyclic networks, input data should not contain" + \
                                         " negative values."
        ranked = True
        for route in flow_routes:
            sort_route = np.sort(route[route > 0])
            ranked = ranked and np.array_equal(sort_route, np.arange(len(sort_route)) + 1)
        assert ranked, "For flow routes from cyclic networks, nodes in each flow route should be ranked from 1 to " + \
                       "n, where n is the number of nodes in the flow route."
    assert flow_profile.ndim == 2, f"Incorrect dimension number ({flow_routes.ndim}) of flow profile. Expect 2."
    assert np.all(flow_profile >= 0), "All the values in flow profile should be non-negative."
    assert flow_routes.shape[0] == flow_profile.shape[
        0], "Inconsistent flow number in network and flow profile detected."
    return flow_routes, (flow_profile, per_hop)


def aggregate_flow(flow_routes, flow_profile):
    """
    Aggregate flows with the same route and deadline class.
    :param flow_routes: the route taken by each flow.
    :param flow_profile: the profile of each flow.
    :return: the aggregated flows.
    """
    # Declare variables to keep the aggregated flow routes and profiles.
    net_agg = np.zeros((0, flow_routes.shape[1]), flow_routes.dtype)
    flow_agg = np.zeros((0, flow_profile.shape[1]), flow_profile.dtype)
    # Find all the flows with the same route.
    net_route = defaultdict(list)
    for idx, route in enumerate(flow_routes):
        net_route[tuple(route)].append(idx)
    for route in net_route:
        route_mask = np.zeros((flow_routes.shape[0],), dtype=bool)
        route_mask[net_route[route]] = True
        net_route[route] = route_mask
    # Find all flows with the same route and deadline class.
    ddl_class = np.unique(flow_profile[:, 2])
    for route in net_route:
        for ddl in ddl_class:
            route_mask = net_route[route]
            ddl_mask = flow_profile[:, 2] == ddl
            route_ddl_mask = np.logical_and(route_mask, ddl_mask)
            # Aggregate the flows.
            if np.sum(route_ddl_mask) > 0:
                ddl_flow = flow_profile[route_ddl_mask]
                ddl_flow = np.sum(ddl_flow, axis=0, keepdims=True)
                ddl_flow[:, 2] = ddl
                net_agg = np.concatenate((net_agg, np.array([route])), axis=0)
                flow_agg = np.concatenate((flow_agg, ddl_flow), axis=0)
    return net_agg, flow_agg


def load_weight(objective, weight_path, num_link):
    """
    Load bandwidth weight and do sanity check.
    :param objective: the objective function.
    :param weight_path: path to the weight profile.
    :param num_link: number of link in the network.
    :return: the loaded weight.
    """
    assert any([objective == obj for obj in [0, 1, 2]]), "Objective function (--objective) must be one of 0 " + \
                                                         "(total link bandwidth), 1 (weighted total " + \
                                                         "link bandwidth), or 2 (maximum link bandwidth)."
    weight = np.ones((num_link,), dtype=float)
    if objective == 1 and weight_path != "":
        weight = np.load(weight_path)
        assert np.all(weight >= 0), f"All the bandwidth weights should be non-negative."
        assert weight.ndim == 1, f"Incorrect dimension number ({weight.ndim}) for weight profile. Expect 1."
        assert len(weight) == num_link, "Inconsistent link number in network and weight profile detected."
    return weight


def check_mode(mode):
    """
    Check the execution mode.
    :param mode: the execution mode.
    """
    assert any([mode == m for m in [0, 1]]), "Execution mode (--mode) must be either 0 (accurate) or 1 (greedy)."
    return mode


def add_set(order_set, item):
    """Add an item to set and return if the item is added successfully or not."""
    num_item = len(order_set)
    order_set.add(item)
    return len(order_set) != num_item


def newton_method(func, der, x, tolerance, log=True):
    """
    Implementation of Newton's method to find the foot of a function.
    Caveat: the caller is responsible for ensuring that the derivative is computed correctly.
    :param func: the function.
    :param der: the derivative of the function.
    :param x: the initial value to start with.
    :param tolerance: the error tolerance.
    :param log: if function includes log(x) term.
    :return: the root.
    """
    while True:
        x_new = x - func(x) / der(x)
        if log:
            x_new = x_new if x_new > 0 else 1
        if abs(x_new - x) < tolerance:
            break
        x = x_new
    return x
