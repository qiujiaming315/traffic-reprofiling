from collections import defaultdict
import numpy as np

"""Utility functions."""


def load_net(net_path, flow_path, aggregate=False):
    """
    Load data and do sanity check.
    :param net_path: path to the network profile (topology and flow routes).
    :param flow_path: path to the flow profile.
    :param aggregate: whether the flows with same route and deadline class should be aggregated.
    :return: the loaded data.
    """
    net_topology, flow_data = np.load(net_path), np.load(flow_path)
    flow_profile, per_hop = flow_data['flow'], flow_data['per_hop']
    if aggregate:
        net_topology, flow_profile = aggregate_flow(net_topology, flow_profile)
    net_type = net_topology.dtype
    assert net_type is np.dtype(bool) or np.issubdtype(net_type, np.integer), f"Incorrect data type ({net_type}) " + \
                                                                              "for network profile. Expect bool or int."
    assert net_topology.ndim == 2, f"Incorrect dimension number ({net_topology.ndim}) for network profile. Expect 2."
    assert np.all(np.sum(net_topology > 0,
                         axis=1) >= 2), "Each flow in the network profile should pass through at least 2 nodes."
    if net_type is np.dtype(int):
        assert np.all(net_topology >= 0), "For cyclic network profiles, input data should not contain negative values."
        ranked = True
        for route in net_topology:
            sort_route = np.sort(route[route > 0])
            ranked = ranked and np.array_equal(sort_route, np.arange(len(sort_route)) + 1)
        assert ranked, "For cyclic network profiles, nodes in each flow route should be ranked from 1 to n, " + \
                       "where n is the number of nodes in the flow route."
    assert flow_profile.ndim == 2, f"Incorrect dimension number ({net_topology.ndim}) of flow profile. Expect 2."
    assert np.all(flow_profile >= 0), "All the values in flow profile should be non-negative."
    assert net_topology.shape[0] == flow_profile.shape[
        0], "Inconsistent flow number in network and flow profile detected."
    return net_topology, (flow_profile, per_hop)


def aggregate_flow(net_topology, flow_profile):
    """
    Aggregate flows with the same route and deadline class.
    :param net_topology: the route taken by each flow.
    :param flow_profile: the profile of each flow.
    :return: the aggregated flows.
    """
    # Declare variables to keep the aggregated flow routes and profiles.
    net_agg = np.zeros((0, net_topology.shape[1]), net_topology.dtype)
    flow_agg = np.zeros((0, flow_profile.shape[1]), flow_profile.dtype)
    # Find all the flows with the same route.
    net_route = defaultdict(list)
    for idx, route in enumerate(net_topology):
        net_route[tuple(route)].append(idx)
    for route in net_route:
        route_mask = np.zeros((net_topology.shape[0],), dtype=bool)
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
