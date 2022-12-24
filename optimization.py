import numpy as np
import argparse
import re
import time
from pathlib import Path

from lib.utils import load_net, load_weight, check_mode
import lib.network_parser as net_parser
from lib.genetic_twoSlope import GATwoSlopeOuter

"""
Calculate minimum required total bandwidth for a given network (flow path + profile)
through intelligent traffic reprofiling using SCED schedulers at every hop.
"""


def main(opts):
    start = time.time()
    # Load the network and flow profile.
    net_profile, (flow_profile, per_hop) = load_net(opts.net, opts.flow, opts.aggregate)
    # Parse the input data and compute the minimum bandwidth of baseline solutions.
    path_matrix = net_parser.parse_link(net_profile)
    if per_hop:
        # Multiply end-to-end deadline by hop count when the loaded deadlines are specified as "per-hop".
        flow_profile[:, 2] = flow_profile[:, 2] * np.sum(path_matrix, axis=1)
    weight = load_weight(opts.objective, opts.weight, path_matrix.shape[1])
    rpps_solution = net_parser.rate_proportional(path_matrix, flow_profile, opts.objective, weight)
    fr_solution = net_parser.full_reprofiling(path_matrix, flow_profile, opts.objective, weight)
    nr_solution = net_parser.no_reprofiling(path_matrix, flow_profile, opts.objective, weight)
    result = dict(path_matrix=path_matrix, rpps=rpps_solution, fr=fr_solution, nr=nr_solution)
    # Determine the execution mode.
    mode = check_mode(opts.mode)
    if mode == 1:
        num_workers = None if opts.num_workers <= 0 else opts.num_workers
        best_solution, best_reprofiling, best_ddl, _ = net_parser.greedy(path_matrix, flow_profile, opts.objective,
                                                                         weight, k=opts.k, num_iter=opts.num_iter,
                                                                         num_workers=num_workers)
    else:
        # Run genetic algorithm to find a good ordering of flow per-hop deadlines.
        genetic = GATwoSlopeOuter(path_matrix, flow_profile, opts.objective, weight)
        best_solution, best_var, _ = genetic.evolve()
        best_reprofiling, best_ddl, _ = net_parser.parse_solution(path_matrix, best_var)
        # Uncomment the following code snippet if you want to perform sanity check on the solution.
        # check = net_parser.check_solution(path_matrix, flow_profile, best_var)
        # if check:
        #     print("Pass Sanity Check.")
    end = time.time()
    print(f"Best solution found: {best_solution:.2f}.")
    print(f"Rate-proportional solution: {rpps_solution:.2f}.")
    print(f"Full reprofiling solution: {fr_solution:.2f}.")
    print(f"No reprofiling solution: {nr_solution:.2f}.")
    print(f"Algorithm execution time: {end - start:.1f}s")
    for key, value in zip(["solution", "reprofiling_delay", "ddl", "run_time"],
                          [best_solution, best_reprofiling, best_ddl, end - start]):
        result[key] = value
    # Save the results to the specified directory.
    net_idx = re.match(r"net(\d+)\.npy", opts.net.replace('\\', '/').split('/')[-1]).group(1)
    flow_idx = re.match(r"flow(\d+)\.npz", opts.flow.replace('\\', '/').split('/')[-1]).group(1)
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    np.savez(opts.out + f"/{path_matrix.shape[0]}_{net_idx}_{flow_idx}.npz", **result)
    return


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('net', help="Path to the input npy file describing network topology and flow routes.")
    args.add_argument('flow', help="Path to the input npz file describing flow profiles.")
    args.add_argument('out', help="Directory to save results.")
    args.add_argument('--objective', type=int, default=0,
                      help="Type of objective function to minimize. 0 for total link bandwidth, " +
                           "1 for weighted total link bandwidth, 2 for maximum link bandwidth.")
    args.add_argument('--weight', type=str, default="",
                      help="Path to the bandwidth weights. Only needed when objective function is weighted " +
                           "total link bandwidth. Weight each link equally by default.")
    args.add_argument('--mode', type=int, default=1,
                      help="Run in accurate (0, solve multiple non-linear programs and find the best result), " +
                           "or greedy (1, a heuristic-based greedy algorithm) mode.")
    args.add_argument('--aggregate', action="store_true",
                      help="Whether flows with same route and deadline class should be aggregated.")
    args.add_argument('--num_workers', type=int, default=0,
                      help="The number of workers for parallel computing. Currently only available for the greedy" +
                           "algorithm. If a non-positive value is given, use the all the processors.")
    args.add_argument('--k', type=int, default=4,
                      help="Number of (intermediate) initial solutions for the greedy algorithm to explore in each " +
                           "iteration of importance sampling. The best result is returned.")
    args.add_argument('--num_iter', type=int, default=2,
                      help="Number of importance sampling iterations for the greedy algorithm.")
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
