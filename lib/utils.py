import numpy as np

"""Utility functions."""


def load_net(net_path, flow_path):
    """
    Load data and do sanity check.
    :param net_path: location of network profile (topology and flow routes).
    :param flow_path: location of flow profile.
    :return: the loaded data.
    """
    net_topology, flow_data = np.load(net_path), np.load(flow_path)
    flow_profile, flow_hop = flow_data['flow'], flow_data['per_hop']
    net_type = net_topology.dtype
    assert net_type is np.dtype(bool) or net_type is np.dtype(
        int), f"Incorrect data type ({net_type}) for network profile. Expect bool or int."
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
    return net_topology, (flow_profile, flow_hop)


def load_weight(objective, weight_path, num_link):
    """
    Load bandwidth weight and do sanity check.
    :param objective: the objective function.
    :param weight_path: location of weight profile.
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
    assert any([mode == m for m in [0, 1, 2]]), "Execution mode (--fast) must be one of 0 (accurate mode)" + \
                                                ", 1 (fast mode), or 2 (greedy mode)."
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
