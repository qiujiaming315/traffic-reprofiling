import numpy as np
import argparse
import re
import time
from pathlib import Path
import os

from lib.utils import load_net, load_weight
import lib.network_parser as net_parser
from lib.genetic_oneSlope import GAOneSlope
from lib.genetic_twoSlope import GATwoSlopeOuter

"""
Calculate minimum required total bandwidth for a given network (topology+routing)
using EDF(SCED)/static priority/FIFO scheduler and per-hop traffic (re)shaper.
"""


def main(opts):

    # path = "output/two_slope/sum/fast/"
    # data = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # err = 1e-4
    #
    # for npz in data:
    #     npz_info = npz[:-4].split('_')
    #     num_flow, net_idx, flow_idx = int(npz_info[0]), int(npz_info[2]), int(npz_info[3])
    #     net_profile, flow_profile = load_net(f"input/network/feed_forward/random/{num_flow}/net{net_idx}.npy",
    #                                          f"input/flow/random/{num_flow}/flow{flow_idx}.npy")
    #     route = net_parser.parse_link(net_profile)
    #     content = np.load(path + npz, allow_pickle=True)
    #     original_bandwidth = content['solution']
    #     solution = content['var'].item()
    #     new_bandwidth = net_parser.improve_solution(route, flow_profile, solution, opts.objective, [], opts.two_slope)
    #     assert new_bandwidth < original_bandwidth + err
    #     if new_bandwidth < original_bandwidth - err:
    #         print("wtf")
    # print()

    start = time.time()
    # Load the network and flow profile.
    # Parse the input data and compute the minimum bandwidth of baseline solutions.
    net_profile, flow_profile = load_net(opts.net, opts.flow)
    route = net_parser.parse_link(net_profile)
    weight = load_weight(opts.objective, opts.weight, route.shape[1])
    rp_solution = net_parser.rate_proportional(route, flow_profile, opts.objective, weight)
    ns_solution = net_parser.no_shaping(route, flow_profile, opts.objective, weight)
    # Run genetic algorithm to find a good ordering of flow.
    genetic = GATwoSlopeOuter(route, flow_profile, opts.objective, weight, opts.fast) if opts.two_slope else GAOneSlope(
        route, flow_profile, opts.objective, weight)
    best_solution, best_var, best_order, opt_list = genetic.evolve()
    # best_solution = net_parser.improve_rate_proportional(route, flow_profile, opts.objective, weight, 10)
    end = time.time()
    print(f"Rate-proportional solution: {rp_solution:.2f}.")
    print(f"No reshaping solution: {ns_solution:.2f}.")
    print(f"Best solution found for EDF: {best_solution:.2f}.")
    print(f"Algorithm execution time: {end - start:.1f}s")

    # Uncomment the following code snippet if you want to perform sanity check on the solution.
    # check = net_parser.check_solution(route, flow_profile, best_var, opts.two_slope)
    # if check:
    #     print("Pass Sanity Check.")

    net_idx = re.match(r"net(\d+)\.npy", opts.net.split('/')[-1]).group(1)
    flow_idx = re.match(r"flow(\d+)\.npy", opts.flow.split('/')[-1]).group(1)
    dir_name = "/two_slope" if opts.two_slope else "/one_slope"
    dir_name += ["/sum", "/weight", "/max"][opts.objective]
    if opts.two_slope:
        dir_name += "/fast" if opts.fast else "/accurate"

    Path(opts.out + dir_name).mkdir(parents=True, exist_ok=True)
    np.savez(opts.out + dir_name + f"/{route.shape[0]}_{route.shape[1]}_{net_idx}_{flow_idx}.npz",
             solution=best_solution, rp=rp_solution, ns=ns_solution, var=best_var, opt_list=opt_list, time=end - start)
    return


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('net', help="Path to the input npy file describing network topology and flow routes.")
    args.add_argument('flow', help="Path to the input npy file describing flow profiles.")
    args.add_argument('out', help="Directory to save results.")
    args.add_argument('--objective', type=int, default=0,
                      help="Type of objective function to minimize. 0 for total link bandwidth, " +
                           "1 for weighted total link bandwidth, 2 for maximum link bandwidth.")
    args.add_argument('--weight', type=str, default="",
                      help="Path to the bandwidth weights. Only needed when objective function is weighted " +
                           "total link bandwidth. Weight each link equally by default.")
    args.add_argument('--two-slope', action='store_true', help="Use one slope (False) or two slope (True) (re)shapers.")
    args.add_argument('--fast', action='store_true',
                      help="Run in accurate (False, solve multiple non-linear programs and find the best result), " +
                           "or fast (True, solve only one non-linear program guaranteed to be no worse than " +
                           "rate-proportional) mode when two slope (re)shapers are applied.")
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
