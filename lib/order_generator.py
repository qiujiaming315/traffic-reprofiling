import numpy as np
from itertools import permutations
from collections import defaultdict


def enum_permutation(array):
    """
    Enumerate all permutations of a given array.
    Caveat: the caller is responsible for ensuring that the size of array is not too large.
    :param array: the array to permute on.
    :return: a 2-D array containing all permutations of the input array.
    """
    permutation = list(permutations(array))
    return np.array(permutation)


def get_partial_order(reprofiling_order, flow_order):
    """
    Collect the partial order given the two orderings.
    :param reprofiling_order: the order of reprofiling delays of all the flows.
    :param flow_order: the order of flow local deadlines at one hop.
    :return: a list containing all the partial order tuples.
    """
    partial_order = list()
    for i in range(len(flow_order)):
        for j in range(i + 1, len(flow_order)):
            if reprofiling_order[flow_order[i]] < reprofiling_order[flow_order[j]]:
                partial_order.append((flow_order[i], flow_order[j]))
    return partial_order


def rand_topological_sort(partial_order):
    """
    Perform topological sort to generate a random total order of elements given the partial order.
    Caveat: the caller is responsible for ensuring that the elements to be sorted are hashable.
    :param partial_order: a list of tuples describing pairwise relationship between elements.
    :return: a list containing all elements in a random feasible total order, or an empty list if infeasible.
    """
    graph, in_degree, nodes = defaultdict(list), defaultdict(int), set()
    queue, order = list(), list()
    # Construct graph and record the in degree of each node.
    for edge in partial_order:
        graph[edge[0]].append(edge[1])
        in_degree[edge[1]] += 1
        nodes.add(edge[0])
        nodes.add(edge[1])
    # Put elements with 0 in degree into the queue.
    for node in nodes:
        if in_degree[node] == 0:
            queue.append(node)
    while len(queue):
        # Randomly pop an node from the queue and add it to the ordered list.
        u = queue.pop(np.random.randint(len(queue)))
        order.append(u)
        # Update the in degree of nodes and add new nodes with 0 in degree to the queue.
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    # Check the in degree dictionary for feasibility.
    for (key, value) in in_degree.items():
        if value:
            return list()
    return order


def enum_topological_sort(partial_order, upper_bound):
    """
    Perform topological sort and generate all feasible total orders given partial order.
    Caveat: the caller is responsible for ensuring that the nodes to be sorted are hashable.
    :param partial_order: a list of tuples describing pairwise relationship between nodes.
    :param upper_bound: an upper bound for the total orders.
    :return: a list containing all (or the upper bound number of) feasible total orders, or an empty list if infeasible.
    """
    graph, in_degree, nodes = defaultdict(list), defaultdict(int), set()
    queue, result, order = list(), list(), list()
    # Construct graph and record the in degree of each node.
    for edge in partial_order:
        graph[edge[0]].append(edge[1])
        in_degree[edge[1]] += 1
        nodes.add(edge[0])
        nodes.add(edge[1])
    # Put nodes with 0 in degree into the queue.
    for node in nodes:
        if in_degree[node] == 0:
            queue.append(node)

    def traverse_graph():
        # Pop and traverse each node from the queue in order.
        num_node = len(queue)
        if num_node == 0:
            # Check the in degree dictionary for feasibility.
            for (key, value) in in_degree.items():
                if value:
                    return
            result.append(order.copy())
            return
        for idx in range(num_node):
            u = queue.pop(idx)
            order.append(u)
            # Update the in degree of nodes and add new nodes with 0 in degree to the queue.
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
            traverse_graph()
            # Return early if the graph is cyclic and topological sorting is infeasible.
            if len(result) == 0 or len(result) > upper_bound:
                return
            # Restore the queue for backtracking purpose.
            for v in graph[u]:
                if in_degree[v] == 0:
                    queue.pop()
                in_degree[v] += 1
            order.pop()
            queue.insert(idx, u)
        return

    traverse_graph()
    return result
