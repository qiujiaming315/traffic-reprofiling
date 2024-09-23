from dataclasses import dataclass

from lib.genetic import GeneticAlgorithm
from lib.genetic_fifo import GATwoSlopeFifo
from lib.utils import add_set
from lib.order_generator import *
from lib.octeract import formulate_sced as octeract_solver
from lib.ipopt import formulate_sced as ipopt_solver
from lib.network_parser import parse_solution
from lib.heuristic_sced import bandwidth_two_slope

"""Implementation of genetic algorithm for optimizing traffic reprofiling under SCED schedulers."""


@dataclass
class OuterOption:
    group_size: int = 5
    survive_size: int = 3
    stable_generation: int = 6
    max_generation: int = float('inf')
    local_size: int = 3
    log_base: float = 1.1
    cross_rate: float = 0.5
    mutation_rate: float = 0.1
    err_tolerance: float = 1e-3
    max_seed: int = 1000


outer_option = OuterOption()


class GATwoSlopeOuter(GATwoSlopeFifo):

    def __init__(self, path_matrix, flow_profile, objective, weight, solver, opts=outer_option):
        self.objective, self.weight = objective, weight
        GeneticAlgorithm.__init__(self, path_matrix, flow_profile, opts)
        self.solver = solver
        self.rp_eval = True

    def evaluate_order(self, order):
        # Retrieve ordering from seed if not in enumeration mode.
        if not self.enum:
            order = np.argsort(order)
        var = GATwoSlopeInner(self.path_matrix, self.flow_profile, order, self.objective, self.weight, self.solver,
                              self.rp_eval)
        solution, _, _ = var.evolve()
        self.rp_eval = False
        update = self.add_opt(solution, var, order)
        return solution, update

    def refine_solution(self, num_refine):
        for idx, (solution, var) in enumerate(zip(self.opt_solution, self.opt_var)):
            if idx < num_refine and var is not None:
                var.opts.max_generation = float('inf')
                if var.max_explored:
                    var.terminate, var.max_explored = False, False
                    best_solution, _, _ = var.evolve()
                    if best_solution < solution:
                        self.opt_solution[idx] = best_solution
        return

    def construct_set(self):
        order_set = enum_permutation(np.arange(self.num_flow))
        np.random.shuffle(order_set)
        # Put the ordering that covers the rate-proportional solution at the top of the list.
        # Ensure that the final result is no worse than rate-proportional by exploring the corresponding ordering first.
        rp_order = self.get_rp_order()
        for idx, order in enumerate(order_set):
            if np.array_equal(order, rp_order):
                order_set[idx] = order_set[0]
                order_set[0] = rp_order
                break
        self.order_set = order_set
        return

    def check_solution(self, var):
        return True

    def get_optimal(self):
        # Return the best solution of the best inner genetic algorithm instance.
        best_idx = np.argmin(self.opt_solution)
        best_var = self.opt_var[best_idx]
        return best_var.get_optimal()

    def add_set_unique(self, order):
        return add_set(self.order_set, tuple(order))


@dataclass
class InnerOption:
    link_thresh: int = 100
    group_size: int = 100
    survive_size: int = 50
    stable_generation: int = 6
    max_generation: int = 3
    local_size: int = 6
    log_base: float = 1.01
    cross_rate: float = 0.2
    mutation_rate: float = 0.1
    err_tolerance: float = 1e-3
    max_seed: int = 10000


inner_option = InnerOption()


class GATwoSlopeInner(GeneticAlgorithm):

    def __init__(self, path_matrix, flow_profile, reprofiling_order, objective, weight, solver, rp=False,
                 opts=inner_option):
        self.local_order = [None] * path_matrix.shape[1]
        self.reprofiling_order, self.reprofiling_dict = reprofiling_order, dict()
        for i, flow_idx in enumerate(reprofiling_order):
            self.reprofiling_dict[flow_idx] = i
        self.rp = rp
        super().__init__(path_matrix, flow_profile, opts)
        nlp_solver = octeract_solver if solver == 0 else ipopt_solver
        self.solver = nlp_solver(path_matrix, flow_profile, objective, weight)
        return

    def initiate(self):
        # Select a subset (for enumeration) from the domain of flow orderings when the size of domain is small.
        if self.enum:
            max_num = min(len(self.order_set), self.opts.group_size)
            self.group = self.order_set[:max_num]
            return
        # Select an initial set of orderings randomly when the domain is large.
        add_num, init_order = 0, list()
        if self.rp:
            # Make sure the rate-proportional ordering is the first one to be explored.
            rp_order = self.get_rp_order()
            add_set(self.order_set, tuple(rp_order))
            init_order.append(rp_order)
            add_num += 1
        num_seed_explored = 0
        while add_num < self.opts.group_size:
            flow_order = np.zeros_like(self.mask)
            for link_idx in range(self.num_link):
                flow_order[self.mask == link_idx] = self.rand_local_order(link_idx)
            if add_set(self.order_set, tuple(flow_order)):
                init_order.append(flow_order)
                add_num += 1
                num_seed_explored = 0
            else:
                num_seed_explored += 1
                if num_seed_explored >= self.opts.max_seed:
                    # Claim no other unique seed exists and stop searching.
                    self.ub = len(self.order_set)
                    break
        self.group = np.array(init_order)
        return

    def mutate(self):
        # Modify the orderings by selecting some hop probabilistically,
        # and then replacing the local ordering at this hop with a random ordering.
        # Do not modify the flow orderings in enumeration mode.
        if self.enum:
            return
        # Modify the flow orderings from crossover and return a set of modified orderings.
        add_size = min(self.opts.group_size, self.ub - len(self.order_set))
        add_num, mutated = 0, list()
        num_seed_explored = 0
        terminate = False
        while not terminate:
            for order in self.group:
                mutation_mask = np.random.rand(self.num_link) < self.opts.mutation_rate
                mutation_link = np.arange(self.num_link)[mutation_mask]
                for link_idx in mutation_link:
                    order[self.mask == link_idx] = self.rand_local_order(link_idx)
                if add_set(self.order_set, tuple(order)):
                    mutated.append(order)
                    add_num += 1
                    num_seed_explored = 0
                    if add_num >= add_size:
                        terminate = True
                        break
                else:
                    num_seed_explored += 1
                    if num_seed_explored >= self.opts.max_seed:
                        # Claim no other unique seed exists and stop searching.
                        self.ub = len(self.order_set)
                        terminate = True
                        break
        self.group = np.array(mutated)
        return

    def total_num(self):
        # TODO: Check if different hops have the same set (number) of flows to simplify computation.
        exact_total = True
        for link_idx in range(self.num_link):
            link_flow = np.arange(self.num_flow)[self.path_matrix[:, link_idx]]
            order_list = list()
            if np.sum(np.log(np.arange(len(link_flow)) + 1)) <= np.log(self.opts.link_thresh):
                for flow_order in enum_permutation(link_flow):
                    partial_order = self.set_partial(flow_order)
                    total_order = enum_topological_sort(partial_order, self.opts.link_thresh - len(order_list))
                    total_order = [[f[0] for f in order] for order in total_order]
                    order_list.extend(total_order)
                    if len(order_list) > self.opts.link_thresh:
                        order_list = list()
                        break
            if len(order_list) == 0:
                exact_total = False
                self.total += np.sum(np.log(np.arange(2 * len(link_flow)) + 1)) - len(link_flow) * np.log(2)
            else:
                self.total += np.log(len(order_list))
            self.local_order[link_idx] = np.array(order_list)
        if not exact_total:
            # Use binary search to find an appropriate log base value.
            left, right = 1, 2
            while True:
                thresh = self.get_thresh(3, right)
                if np.log(thresh) < self.total:
                    break
                left = right
                right = 2 * right
            while abs(right - left) > self.opts.err_tolerance:
                mid = left + (right - left) / 2
                thresh = self.get_thresh(3, mid)
                if np.log(thresh) < self.total:
                    right = mid
                else:
                    left = mid
            self.opts.log_base = max(self.opts.log_base, right)
        return

    def set_mask(self):
        cut = np.cumsum(2 * np.sum(self.path_matrix, axis=0))
        mask = np.zeros((cut[-1],), dtype=int)
        mask[cut[:-1]] = 1
        self.mask = np.cumsum(mask)
        self.cut = np.append(0, cut)
        return

    def construct_set(self):
        order_set = list()
        init_order = np.zeros_like(self.mask)

        def enum_order(half_order, cur_idx):
            num_scale = self.local_order[cur_idx].shape[0]
            complete_order = np.ones((num_scale, 1), dtype=int) * half_order
            complete_order[:, self.mask == cur_idx] = self.local_order[cur_idx]
            if cur_idx == self.num_link - 1:
                for order in complete_order:
                    order_set.append(order)
            else:
                for order in complete_order:
                    enum_order(order, cur_idx + 1)

        enum_order(init_order, 0)
        order_set = np.array(order_set)
        np.random.shuffle(order_set)
        # Put the ordering that covers the rate-proportional solution at the top of the list.
        # Ensure that the final result is no worse than rate-proportional by exploring the corresponding ordering first.
        if self.rp:
            rp_order = self.get_rp_order()
            for idx, order in enumerate(order_set):
                if np.array_equal(order, rp_order):
                    order_set[idx] = order_set[0]
                    order_set[0] = rp_order
                    break
        self.order_set = order_set
        return

    def check_solution(self, var):
        tor = 1e-3
        reprofiling_delay, ddl, bandwidth = parse_solution(self.path_matrix, var)
        # Check if solution deadlines are non-negative.
        feasible1 = np.all(reprofiling_delay >= -tor) and np.all(ddl >= -tor)
        # Check if each flow in-network deadline stays in range.
        total_ddl, rd_ub = self.flow_profile[:, 2], self.flow_profile[:, 1] / self.flow_profile[:, 0]
        net_ddl = np.sum(ddl, axis=1)
        feasible2 = np.all(net_ddl + reprofiling_delay <= total_ddl + tor)
        feasible3 = np.all(reprofiling_delay <= rd_ub + tor)
        feasible = feasible1 and feasible2 and feasible3
        # Check if the computed bandwidth is consistent with the returned bandwidth.
        actual_bandwidth = bandwidth_two_slope(self.path_matrix, self.flow_profile, reprofiling_delay, ddl)
        tight = np.all(np.abs(actual_bandwidth - bandwidth) <= tor)
        return feasible and tight

    def set_partial(self, flow_order):
        """
        Helper function to properly set partial order for topological sorting.
        :param flow_order: the order of the flow local deadline at one hop.
        :return: a list of partial orders.
        """
        partial_order = get_partial_order(self.reprofiling_dict, flow_order)
        partial_order = [((f1, 2), (f2, 2)) for (f1, f2) in partial_order]
        partial_order.extend(
            [((flow_order[x], 1), (flow_order[x + 1], 1)) for x in range(len(flow_order) - 1)])
        partial_order.extend([((f, 1), (f, 2)) for f in flow_order])
        return partial_order

    def get_rp_order(self):
        """
        Helper function to generate an ordering (randomly) that covers the rate-proportional solution.
        :return: the generated ordering.
        """
        rp_order = np.zeros_like(self.mask)
        for link_idx in range(self.num_link):
            link_flow = self.reprofiling_order[self.path_matrix[:, link_idx][self.reprofiling_order]]
            rand_order = np.random.permutation(link_flow)
            rand_order = np.concatenate((rand_order, link_flow))
            rp_order[self.mask == link_idx] = rand_order
        return rp_order

    def rand_local_order(self, link_idx):
        """
        Helper function to generate a random feasible ordering at one hop.
        :param link_idx: the hop index.
        :return: a random ordering at this hop.
        """
        num_order = len(self.local_order[link_idx])
        if num_order:
            return self.local_order[link_idx][np.random.randint(num_order)]
        link_flow = np.arange(self.num_flow)[self.path_matrix[:, link_idx]]
        rand_order = np.random.permutation(link_flow)
        partial_order = self.set_partial(rand_order)
        total_order = rand_topological_sort(partial_order)
        total_order = np.array([f[0] for f in total_order])
        return total_order
