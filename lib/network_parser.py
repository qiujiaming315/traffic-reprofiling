import numpy as np
from concurrent.futures import ThreadPoolExecutor as TPE
from itertools import repeat

from lib.heuristic_fifo import improve_two_slope as improve_fifo
from lib.heuristic_sced import improve_two_slope as improve_sced

"""
Functions that help parse the inputs and the solutions of the optimization problem.
Framework of the heuristic-based Greedy algorithm.
"""

SCHEDULER = 0  # the scheduler used at every hop in the network. 0 for FIFO and 1 for SCED.


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


def greedy(path_matrix, flow_profile, objective, weight, k=4, num_iter=2, num_workers=None, min_improvement=0.001):
    """
    Use importance sampling + greedy algorithm to efficiently approximate the optimal solution (guaranteed to be no
    worse than both the full reprofiling solution and the no reprofiling solution).
    Generate initial reprofiling ratios shared by all the flows.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :param k: number of (intermediate) initial solutions to sample (uniformly) from the importance region
        in each iteration.
    :param num_iter: number of iterations to refine the importance region.
    :param num_workers: number of workers for parallel computing.
    :param min_improvement: minimum improvement to terminate iteration.
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
    for iter_idx in range(num_iter):
        # Uniformly sample a range of reprofiling ratios between the two extremes.
        reprofiling_ratios = np.linspace(low_ratio, high_ratio, num=k + 2)
        # Remove the two extremes (because they were explored) if it's not the first iteration.
        if iter_idx > 0:
            reprofiling_ratios = reprofiling_ratios[1:-1]
        # Apply the greedy algorithm to improve each initial solution in parallel.
        with TPE(max_workers=num_workers) as pool:
            solutions = list(
                pool.map(greedy_, repeat(path_matrix), repeat(flow_profile), reprofiling_ratios, repeat(objective),
                         repeat(weight)))
        # Retrieve the best solution in the current iteration.
        solution_sort = sorted(range(len(solutions)), key=lambda x: solutions[x][0])
        best_idx = solution_sort[0]
        improvement = (best_solution - solutions[best_idx][0]) / best_solution if iter_idx > 0 else np.inf
        # Update the best solution and the two extremes of the importance region.
        if solutions[best_idx][0] < best_solution:
            best_solution, best_reprofiling, best_ddl = solutions[best_idx]
            best_ratio = reprofiling_ratios[best_idx]
        low_ratio = reprofiling_ratios[max(0, best_idx - 1)]
        high_ratio = reprofiling_ratios[min(len(reprofiling_ratios) - 1, best_idx + 1)]
        # Terminate the iteration if the improvement is small enough.
        if improvement < min_improvement:
            break
    return best_solution, best_reprofiling, best_ddl, best_ratio


def greedy_(path_matrix, flow_profile, reprofiling_ratio, objective, weight):
    """
    Use the specified reprofiling ratio to generate an initial solution based on even deadline splitting.
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
    # Improve the deadline assignment using the greedy heuristic.
    new_solution, reprofiling_delay, ddl = improve_solution(path_matrix, flow_profile, (reprofiling_delay, ddl),
                                                            objective, weight)
    return new_solution, reprofiling_delay, ddl


def improve_solution(path_matrix, flow_profile, solution, objective, weight, min_improvement=0.001):
    """
    Improve the solution using a heuristic-based iterative algorithm.
    :param path_matrix: the network routes.
    :param flow_profile: the flow profile.
    :param solution: reprofiling delay, per-hop deadline, and bandwidth assigned by a solution.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :param min_improvement: minimum improvement to terminate iteration.
    :return: the improved solution.
    """
    reprofiling_delay, ddl = solution
    prev_bandwidth, initial_solution = np.inf, True
    # Run the greedy re(re)profiling algorithm iteratively until the solution cannot be further improved.
    while True:
        # Improve the solution.
        improve_func = improve_fifo if SCHEDULER == 0 else improve_sced
        reprofiling_delay, ddl, bandwidth = improve_func(path_matrix, flow_profile, reprofiling_delay, ddl)
        current_bandwidth = get_objective(bandwidth, objective, weight)
        # Break when the improvement is marginal.
        improvement = np.inf if initial_solution else (prev_bandwidth - current_bandwidth) / prev_bandwidth
        if improvement < min_improvement:
            break
        prev_bandwidth, initial_solution = current_bandwidth, False
    return current_bandwidth, reprofiling_delay, ddl
