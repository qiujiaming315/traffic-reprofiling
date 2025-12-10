import numpy as np
import argparse
import os
import time
from pathlib import Path

from lib.utils import load_net, load_weight, check_mode
import lib.network_parser as net_parser
import lib.greedy as greedy
import lib.heuristic_fifo as fifo
import lib.heuristic_sced as sced
import lib.heuristic_priority as priority
from lib.genetic_fifo import GATwoSlopeFifo
from lib.genetic_sced import GATwoSlopeOuter

"""
Calculate minimum required total bandwidth for a given network (flow path + profile)
through intelligent traffic reprofiling using SCED/FIFO schedulers at every hop.
"""


def main(opts):
    start = time.time()
    # Load the flow routes and flow profile.
    flow_routes, (flow_profile, per_hop) = load_net(opts.route, opts.flow, opts.aggregate, opts.aggregate_path)
    # Parse the input data and compute the minimum bandwidth of baseline solutions.
    path_matrix, path_order, link_map = net_parser.parse_link(flow_routes)
    if per_hop:
        # Multiply end-to-end deadline by hop count when the loaded deadlines are specified as "per-hop".
        flow_profile[:, 2] = flow_profile[:, 2] * np.sum(path_matrix, axis=1)
    weight = load_weight(opts.objective, opts.weight, path_matrix.shape[1])
    # Set the heuristic and NLP-based algorithms according to the scheduler.
    if opts.scheduler == 0:
        heuristic, nlp_genetic = fifo, GATwoSlopeFifo
    elif opts.scheduler == 1:
        heuristic, nlp_genetic = sced, GATwoSlopeOuter
    else:
        heuristic, nlp_genetic = priority, None
        priority.NUM_CLASS = opts.num_priority_class
    net_parser.SCHEDULER = opts.scheduler
    greedy.SCHEDULER = opts.scheduler
    # Compute the bandwidth requirements of the baseline solutions.
    fr_solution, fr_per_hop, fr_priority = heuristic.full_reprofiling(path_matrix, flow_profile, opts.objective, weight)
    nr_solution, nr_per_hop, nr_ddl, nr_priority = heuristic.no_reprofiling(path_matrix, flow_profile, opts.objective,
                                                                            weight)
    result = dict(path_matrix=path_matrix, link_map=link_map, fr=fr_solution, nr=nr_solution, fr_=fr_per_hop,
                  nr_=nr_per_hop)
    # Determine the execution mode.
    mode = check_mode(opts.mode)
    best_priority = None
    if mode == 1:
        num_workers = None if opts.num_workers <= 0 else opts.num_workers
        best_solution, best_reprofiling, best_ddl, _, best_per_hop, best_priority = greedy.greedy(path_matrix,
                                                                                                  flow_profile,
                                                                                                  opts.objective,
                                                                                                  weight, k=opts.k,
                                                                                                  num_iter=opts.num_iter,
                                                                                                  num_workers=num_workers)
    else:
        assert opts.scheduler != 2, ("(NLP-based) accurate mode currently does not support static priority schedulers. "
                                     "Please run in greedy mode instead.")
        # Run genetic algorithm to find a good ordering of flow per-hop deadlines.
        genetic = nlp_genetic(path_matrix, flow_profile, opts.objective, weight, opts.solver)
        best_solution, best_var, _ = genetic.evolve()
        best_reprofiling, best_ddl, best_per_hop = net_parser.parse_solution(path_matrix, best_var)
    end = time.time()
    print(f"Best solution found: {best_solution:.2f}.")
    print(f"Full reprofiling solution: {fr_solution:.2f}.")
    print(f"No reprofiling solution: {nr_solution:.2f}.")
    print(f"Algorithm execution time: {end - start:.1f}s")
    # Uncomment the following code snippet if you want to retrieve buffer bounds.
    link_buffer, reprofiler_buffer = net_parser.get_buffer_bound(path_matrix, path_order, flow_profile,
                                                                 best_reprofiling, best_ddl, best_per_hop)
    for key, value in zip(["solution", "solution_", "reprofiling_delay", "ddl", "run_time"],
                          [best_solution, best_per_hop, best_reprofiling, best_ddl, end - start]):
        result[key] = value
    if opts.scheduler == 2:
        # result["fr_priority"] = fr_priority
        # result["nr_priority"] = nr_priority
        result["priority"] = best_priority
    # Save the results to the specified directory.
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.out, opts.file_name + ".npz"), **result)
    return


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('route', help="Path to the input npy/npz file describing flow routes.")
    args.add_argument('flow', help="Path to the input npz file describing flow profiles.")
    args.add_argument('out', help="Directory to save results.")
    args.add_argument('file_name', help="Name of the file to save results.")
    args.add_argument('--scheduler', type=int, default=0,
                      help="Type of scheduler applied at each hop of the network. 0 for FIFO schedulers, " +
                           "1 for SCED schedulers, 2 for static priority scheduler.")
    args.add_argument('--objective', type=int, default=0,
                      help="Type of objective function to minimize. 0 for total link bandwidth, " +
                           "1 for weighted total link bandwidth, 2 for maximum link bandwidth.")
    args.add_argument('--weight', type=str, default="",
                      help="Path to the bandwidth weights. Only needed when objective function is weighted " +
                           "total link bandwidth. Weight each link equally by default.")
    args.add_argument('--mode', type=int, default=1,
                      help="Run in accurate (0, solve multiple non-linear programs and find the best result), " +
                           "or greedy (1, a heuristic-based greedy algorithm) mode.")
    args.add_argument('--num-priority-class', type=int, default=8,
                      help="Number of priority class. Only active when static priority is chosen as the scheduler.")
    args.add_argument('--solver', type=int, default=1, help="The NLP solver to solve the optimization. 0 for " +
                                                            "octeract and 1 for ipopt.")
    args.add_argument('--aggregate', action="store_true",
                      help="Whether flows with same route and deadline class should be aggregated.")
    args.add_argument('--aggregate_path', type=str, default="",
                      help="Directory to save the aggregated flow routes and profiles. Only active when --aggregate " +
                           "is true. Only save the data when the path is not an empty string.")
    args.add_argument('--num_workers', type=int, default=0,
                      help="The number of workers for parallel computing. Currently only available for the greedy " +
                           "algorithm. If a non-positive value is given, use the all the processors.")
    args.add_argument('--k', type=int, default=4,
                      help="Number of (intermediate) initial solutions for the greedy algorithm to explore in each " +
                           "iteration of importance sampling. The best result is returned.")
    args.add_argument('--num_iter', type=int, default=2,
                      help="Number of importance sampling iterations for the greedy algorithm.")
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
