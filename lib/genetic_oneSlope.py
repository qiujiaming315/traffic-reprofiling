import numpy as np
from dataclasses import dataclass

from lib.genetic import GeneticAlgorithm
from lib.utils import add_set
from lib.order_generator import random_order, enum_permutation
import lib.octeract as octeract

"""Implementation of genetic algorithm for one-slope service curves (shifted token bucket shapers)."""


@dataclass
class GAOption:
    group_size: int = 100
    survive_size: int = 50
    stable_generation: int = 6
    max_generation: int = float('inf')
    local_size: int = 5
    log_base: float = 1.01
    init_rate: float = 0.3
    cross_rate: float = 0.2
    mutation_rate: float = 0.05
    err_tolerance: float = 1e-3


option = GAOption()


class GAOneSlope(GeneticAlgorithm):

    def __init__(self, route, flow_profile, objective, weight, opts=option):
        super().__init__(route, flow_profile, opts)
        self.solver = octeract.formulate(route, flow_profile, objective, weight)

    def initiate(self):
        # Select a subset (for enumeration) from the domain of flow orderings when the size of domain is small.
        if self.enum:
            max_num = min(len(self.order_set), self.opts.group_size)
            self.group = self.order_set[:max_num]
            return
        # Select an initial set combined with total orderings and random orderings when the domain is large.
        add_num, init_order = 0, list()
        num_total = int(self.opts.group_size * self.opts.init_rate)
        flow_size = np.sum(np.log(np.arange(self.num_flow) + 1))
        # Generate a small set of flow orderings assuming there is a total ordering of the flows.
        if flow_size <= np.log(3 * self.opts.group_size):
            num_total = min(np.math.factorial(self.num_flow), num_total)
            total_orders = enum_permutation(np.arange(self.num_flow))
            np.random.shuffle(total_orders)
            total_orders = total_orders[:num_total]
            for order in total_orders:
                order_mask = self.route[order]
                flow_order = order[:, np.newaxis] * np.ones((self.num_link,), dtype=int)
                flow_order = flow_order.T[order_mask.T]
                self.order_set.add(tuple(flow_order))
                init_order.append(flow_order)
            add_num = num_total
        else:
            while add_num < num_total:
                order = np.random.permutation(self.num_flow)
                order_mask = self.route[order]
                flow_order = order[:, np.newaxis] * np.ones((self.num_link,), dtype=int)
                flow_order = flow_order.T[order_mask.T]
                if add_set(self.order_set, tuple(flow_order)):
                    init_order.append(flow_order)
                    add_num += 1
        # Fill the initial set with random flow orderings.
        while add_num < self.opts.group_size:
            flow_order = random_order(self.route)
            order_mask = self.route[flow_order, np.arange(self.num_link)]
            flow_order = flow_order.T[order_mask.T]
            if add_set(self.order_set, tuple(flow_order)):
                init_order.append(flow_order)
                add_num += 1
        self.group = np.array(init_order)
        return

    def mutate(self):
        # Modify the orderings by selecting some flow deadlines probabilistically,
        # and then swapping their orders with some other flows at the same hop.
        # Do not modify the flow orderings in enumeration mode.
        if self.enum:
            return
        # Modify the flow orderings from crossover and return a set of modified orderings.
        add_size = min(self.opts.group_size, self.ub - len(self.order_set))
        add_num, mutated = 0, list()
        while add_num < add_size:
            for order in self.group:
                mutation_mask = np.random.rand(np.size(self.mask)) < self.opts.mutation_rate
                mutation_idx = np.arange(np.size(self.mask))[mutation_mask]
                mutation_link = self.mask[mutation_mask]
                for m_idx, l_idx in zip(mutation_idx, mutation_link):
                    swap_idx = np.random.randint(self.cut[l_idx], self.cut[l_idx + 1])
                    temp = order[m_idx]
                    order[m_idx] = order[swap_idx]
                    order[swap_idx] = temp
                if add_set(self.order_set, tuple(order)):
                    mutated.append(order)
                    add_num += 1
                    if add_num >= add_size:
                        break
        self.group = np.array(mutated)
        return

    def total_num(self):
        sum_flow = np.sum(self.route, axis=0).astype(int)
        for nflow in sum_flow:
            self.total += np.sum(np.log(np.arange(nflow) + 1))
        return

    def set_mask(self):
        cut = np.cumsum(np.sum(self.route, axis=0))
        mask = np.zeros((cut[-1],), dtype=int)
        mask[cut[:-1]] = 1
        self.mask = np.cumsum(mask)
        self.cut = np.append(0, cut)
        return

    def construct_set(self):
        order_set = list()
        init_order = np.zeros_like(self.mask)
        flow_permutation = list()
        for link_idx in range(self.num_link):
            link_mask = self.route[:, link_idx]
            link_flow = np.arange(self.num_flow)[link_mask]
            flow_permutation.append(enum_permutation(link_flow))

        def enum_order(half_order, cur_idx):
            num_scale = flow_permutation[cur_idx].shape[0]
            complete_order = np.ones((num_scale, 1), dtype=int) * half_order
            complete_order[:, self.mask == cur_idx] = flow_permutation[cur_idx]
            if cur_idx == self.num_link - 1:
                for order in complete_order:
                    order_set.append(order)
            else:
                for order in complete_order:
                    enum_order(order, cur_idx + 1)

        enum_order(init_order, 0)
        order_set = np.array(order_set)
        np.random.shuffle(order_set)
        self.order_set = order_set
        return
