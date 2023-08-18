from dataclasses import dataclass

from lib.genetic import GeneticAlgorithm
from lib.utils import add_set
from lib.order_generator import *
from lib.octeract import formulate_fifo

"""Implementation of genetic algorithm for optimizing traffic reprofiling under FIFO schedulers."""


@dataclass
class FifoOption:
    group_size: int = 100
    survive_size: int = 50
    stable_generation: int = 3
    max_generation: int = float('inf')
    local_size: int = 6
    log_base: float = 1.1
    cross_rate: float = 0.5
    mutation_rate: float = 0.1
    err_tolerance: float = 1e-3


fifo_option = FifoOption()


class GATwoSlopeFifo(GeneticAlgorithm):

    def __init__(self, path_matrix, flow_profile, objective, weight, opts=fifo_option):
        self.objective, self.weight = objective, weight
        super().__init__(path_matrix, flow_profile, opts)
        self.solver = formulate_fifo(path_matrix, flow_profile, objective, weight)

    def initiate(self):
        # Select a subset (for enumeration) from the domain of flow orderings when the size of domain is small.
        if self.enum:
            max_num = min(len(self.order_set), self.opts.group_size)
            self.group = self.order_set[:max_num]
            return
        # Select an initial group of random ordering when the size of domain is large.
        # Make sure the rate-proportional ordering is the first one to be explored.
        add_num, init_order = 1, list()
        rp_order = self.get_rp_order()
        self.add_set_unique(rp_order)
        rp_seed = np.zeros_like(rp_order, dtype=float)
        rp_seed[rp_order] = np.arange(self.num_flow) / self.num_flow
        init_order.append(rp_seed)
        # Fill the initial set with random flow orderings.
        while add_num < self.opts.group_size:
            seed = np.random.rand(self.num_flow)
            if self.add_set_unique(np.argsort(seed)):
                init_order.append(seed)
                add_num += 1
        self.group = np.array(init_order)
        return

    def evaluate_order(self, order):
        # Retrieve ordering from seed if not in enumeration mode.
        if not self.enum:
            order = np.argsort(order)
        solution, var = self.solver(order)
        update = self.add_opt(solution, var, order)
        return solution, update

    def cross_parent(self, p1, p2):
        # Exchange the value at positions selected probabilistically for each pair of parents.
        exchange_mask = np.random.rand(self.num_flow) < self.opts.cross_rate
        c1 = np.where(exchange_mask, p1, p2)
        c2 = np.where(exchange_mask, p2, p1)
        return c1, c2

    def mutate(self):
        # Modify the orderings by selecting some positions probabilistically and change their values.
        # Do not modify the flow orderings in enumeration mode.
        if self.enum:
            return
        # Modify the flow orderings from crossover and return a set of modified orderings.
        add_size = min(self.opts.group_size, self.ub - len(self.order_set))
        add_num, mutated = 0, list()
        while add_num < add_size:
            for seed in self.group:
                mutation_mask = np.random.rand(self.num_flow) < self.opts.mutation_rate
                seed = np.where(mutation_mask, np.random.rand(self.num_flow), seed)
                if self.add_set_unique(np.argsort(seed)):
                    mutated.append(seed)
                    add_num += 1
                    if add_num >= add_size:
                        break
        self.group = np.array(mutated)
        return

    def total_num(self):
        self.total = np.sum(np.log(np.arange(self.num_flow) + 1))
        return

    def construct_set(self):
        order_set_raw = enum_permutation(np.arange(self.num_flow))
        np.random.shuffle(order_set_raw)
        # Put the ordering that covers the rate-proportional solution at the top of the list.
        # Ensure that the final result is no worse than rate-proportional by exploring the corresponding ordering first.
        rp_order = self.get_rp_order()
        order_set = [rp_order]
        self.add_set_unique(rp_order)
        for order in order_set_raw:
            if self.add_set_unique(order):
                order_set.append(order)
        self.order_set = np.array(order_set)
        self.ub = min(self.ub, len(self.order_set))
        return

    def get_rp_order(self):
        """
        Helper function to generate an reprofiling delay ordering that covers the rate-proportional solution.
        :return: the generated ordering.
        """
        max_reprofiling = np.concatenate((self.flow_profile[:, 2][np.newaxis, :],
                                          self.flow_profile[:, 1][np.newaxis, :] / self.flow_profile[:, 0][np.newaxis,
                                                                                   :]), axis=0)
        rp_order = np.argsort(np.amin(max_reprofiling, axis=0))
        return rp_order

    def add_set_unique(self, order):
        """Add a unique flow ordering to set."""
        order_code = np.array([], dtype=int)
        for link_idx in range(self.num_link):
            link_mask = self.path_matrix[:, link_idx]
            order_code = np.append(order_code, order[link_mask[order]])
        return add_set(self.order_set, tuple(order_code))
