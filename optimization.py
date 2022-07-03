import numpy as np
import argparse
import re
import time
from pathlib import Path

from lib.utils import load_net, load_weight, check_mode
import lib.network_parser as net_parser
from lib.genetic_oneSlope import GAOneSlope
from lib.genetic_twoSlope import GATwoSlopeOuter

"""
Calculate minimum required total bandwidth for a given network (flow path + profile)
through intelligent traffic reprofiling using SCED schedulers at every hop.
"""


def main(opts):
    start = time.time()
    # Load the network and flow profile.
    # Parse the input data and compute the minimum bandwidth of baseline solutions.
    net_profile, (flow_profile, flow_hop) = load_net(opts.net, opts.flow)
    route = net_parser.parse_link(net_profile)
    if flow_hop:
        flow_profile[:, 2] = flow_profile[:, 2] * np.sum(route, axis=1)
    weight = load_weight(opts.objective, opts.weight, route.shape[1])
    rp_solution = net_parser.rate_proportional(route, flow_profile, opts.objective, weight)
    ns_solution = net_parser.no_shaping(route, flow_profile, opts.objective, weight)
    result = dict(route=route, rp=rp_solution, ns=ns_solution)
    # Determine the execution mode.
    mode = check_mode(opts.fast)
    if mode == 2:
        best_solution, best_shaping, best_ddl = net_parser.greedy(route, flow_profile, opts.objective, weight, 10)
    else:
        fast = mode == 1
        # Run genetic algorithm to find a good ordering of flow.
        genetic = GATwoSlopeOuter(route, flow_profile, opts.objective, weight, fast) if opts.two_slope else GAOneSlope(
            route, flow_profile, opts.objective, weight)
        best_solution, best_var, best_order, opt_list = genetic.evolve()
        best_shaping, best_ddl, _ = net_parser.parse_solution(route, best_var)
        result["opt_list"] = opt_list
        # Uncomment the following code snippet if you want to perform sanity check on the solution.
        # check = net_parser.check_solution(route, flow_profile, best_var, opts.two_slope)
        # if check:
        #     print("Pass Sanity Check.")
    end = time.time()
    print(f"Best solution found for EDF: {best_solution:.2f}.")
    print(f"Rate-proportional solution: {rp_solution:.2f}.")
    print(f"No reshaping solution: {ns_solution:.2f}.")
    print(f"Algorithm execution time: {end - start:.1f}s")
    for key, value in zip(["solution", "shaping", "ddl", "time"], [best_solution, best_shaping, best_ddl, end - start]):
        result[key] = value

    net_idx = re.match(r"net(\d+)\.npy", opts.net.split('/')[-1]).group(1)
    flow_idx = re.match(r"flow(\d+)\.npz", opts.flow.split('/')[-1]).group(1)
    dir_name = "/two_slope" if opts.two_slope else "/one_slope"
    dir_name += ["/sum", "/weight", "/max"][opts.objective]
    if opts.two_slope:
        dir_name += ["/accurate", "/fast", "/greedy"][opts.fast]

    Path(opts.out + dir_name).mkdir(parents=True, exist_ok=True)
    np.savez(opts.out + dir_name + f"/{route.shape[0]}_{route.shape[1]}_{net_idx}_{flow_idx}.npz", **result)
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
    args.add_argument('--two-slope', action='store_true', help="Use one slope (False) or two slope (True) (re)shapers.")
    args.add_argument('--fast', type=int, default=0,
                      help="Run in accurate (0, solve multiple non-linear programs and find the best result), " +
                           "fast (1, solve only one non-linear program guaranteed to be no worse than " +
                           "rate-proportional), or greedy (2, a greedy algorithm based on heuristic which is " +
                           "the fastest among all three) mode when two slope (re)shapers are applied.")
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
