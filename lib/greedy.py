import numpy as np
from concurrent.futures import ThreadPoolExecutor as TPE
from itertools import repeat

from lib.network_parser import get_objective
from lib.heuristic_fifo import improve_two_slope as improve_fifo
from lib.heuristic_sced import improve_two_slope as improve_sced

"""
Framework of the heuristic-based Greedy algorithm.
"""

SCHEDULER = 0  # the scheduler used at every hop in the network. 0 for FIFO and 1 for SCED.


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
    best_solution, best_reprofiling, best_ddl, best_bandwidth, best_ratio = np.inf, None, None, None, None
    # Specify the two extremes of the importance region.
    low_ratio, high_ratio = 0, 1
    for iter_idx in range(num_iter):
        # Uniformly sample a range of reprofiling ratios between the two extremes.
        reprofiling_ratios = np.linspace(low_ratio, high_ratio, num=k + 2)
        # Remove the two extremes (because they were explored) if it's not the first iteration.
        if iter_idx > 0:
            reprofiling_ratios = reprofiling_ratios[1:-1]
        # Apply the greedy algorithm to improve each initial solution in parallel.
        # with TPE(max_workers=num_workers) as pool:
        #     solutions = list(
        #         pool.map(greedy_, repeat(path_matrix), repeat(flow_profile), reprofiling_ratios, repeat(objective),
        #                  repeat(weight)))

        solutions = list()
        for reprofiling_ratio in reprofiling_ratios:
            solutions.append(greedy_(path_matrix, flow_profile, reprofiling_ratio, objective, weight))

        # Retrieve the best solution in the current iteration.
        solution_sort = sorted(range(len(solutions)), key=lambda x: solutions[x][0])
        best_idx = solution_sort[0]
        improvement = (best_solution - solutions[best_idx][0]) / best_solution if iter_idx > 0 else np.inf
        # Update the best solution and the two extremes of the importance region.
        if solutions[best_idx][0] < best_solution:
            best_solution, best_reprofiling, best_ddl, best_bandwidth = solutions[best_idx]
            best_ratio = reprofiling_ratios[best_idx]
        low_ratio = reprofiling_ratios[max(0, best_idx - 1)]
        high_ratio = reprofiling_ratios[min(len(reprofiling_ratios) - 1, best_idx + 1)]
        # Terminate the iteration if the improvement is small enough.
        if improvement < min_improvement:
            break
    return best_solution, best_reprofiling, best_ddl, best_ratio, best_bandwidth


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
    new_solution, reprofiling_delay, ddl, new_bandwidth = improve_solution(path_matrix, flow_profile,
                                                                           (reprofiling_delay, ddl),
                                                                           objective, weight)
    return new_solution, reprofiling_delay, ddl, new_bandwidth


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
    improve_func = improve_fifo if SCHEDULER == 0 else improve_sced
    while True:
        # Improve the solution.
        reprofiling_delay, ddl, bandwidth = improve_func(path_matrix, flow_profile, reprofiling_delay, ddl)
        current_bandwidth = get_objective(bandwidth, objective, weight)
        # Break when the improvement is marginal.
        improvement = np.inf if initial_solution else (prev_bandwidth - current_bandwidth) / prev_bandwidth
        if improvement < min_improvement:
            break
        prev_bandwidth, initial_solution = current_bandwidth, False
    return current_bandwidth, reprofiling_delay, ddl, bandwidth
