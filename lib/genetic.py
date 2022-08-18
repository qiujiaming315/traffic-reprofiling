import numpy as np

from lib.utils import newton_method

"""Parent class for genetic algorithm."""


class GeneticAlgorithm:

    def __init__(self, path_matrix, flow_profile, opts):
        self.path_matrix, self.flow_profile, self.opts = path_matrix, flow_profile, opts
        self.num_flow, self.num_link = path_matrix.shape
        self.check_opts()
        self.terminate, self.max_explored = False, False
        self.generation, self.stable = 0, 0
        self.group, self.survivor, self.score = None, None, None
        self.total, self.ub, self.enum = 0, 0, False
        self.mask, self.cut = None, None
        self.order_set = set()
        self.total_num()
        self.set_bounds()
        self.opt_solution = np.ones((opts.local_size,)) * np.inf
        self.opt_var, self.opt_order = [None] * opts.local_size, [None] * opts.local_size
        self.set_mask()
        if self.enum:
            self.construct_set()
        self.initiate()

    def evolve(self):
        """Evolve until the termination condition is satisfied and return the optimal solution."""
        while not self.terminate:
            self.evaluate()
            self.crossover()
            self.mutate()
        return self.get_optimal()

    def initiate(self):
        """Generate an initial group of orderings."""
        return

    def evaluate(self):
        """
        Evaluate the fitness of a group of orderings and select the survivors.
        Check whether termination condition is satisfied.
        """
        # Evaluate each ordering and collect the results.
        update, group_score = False, list()
        for order in self.group:
            solution, ud = self.evaluate_order(order)
            update = update or ud
            group_score.append(solution)
        group_score = np.array(group_score)
        # Evaluate the results and keep the candidates with best solutions.
        if self.survivor is not None:
            self.group = np.concatenate((self.survivor, self.group), axis=0)
            group_score = np.concatenate((self.score, group_score))
        sort_idx = np.argsort(group_score)[:self.opts.survive_size]
        self.survivor, self.score = self.group[sort_idx], group_score[sort_idx]
        # Update states and check if termination condition is satisfied.
        self.generation += 1
        self.stable = 0 if update else self.stable + 1
        self.terminate, multi_opt = True, False
        if np.amax(self.opt_solution) - np.amin(self.opt_solution) < self.opts.err_tolerance:
            multi_opt = True
            print("Terminating: Same optimal solutions observed enough times.")
        elif self.generation * self.opts.group_size >= self.ub:
            print("Terminating: Maximum number of trails explored.")
        elif self.stable > self.opts.stable_generation:
            print("Terminating: Explored long enough without improvement.")
        elif self.generation >= self.opts.max_generation:
            self.max_explored = True
            print("Terminating: Maximum number of generations explored.")
        else:
            self.terminate = False
        # Improve the best set of local_solve solutions with global_solve solutions.
        if self.terminate:
            num_refine = 1 if multi_opt else self.opts.local_size
            self.refine_solution(num_refine)
            print(f"Evolution lasts {self.generation} generations. " +
                  f"Observe no improvement for {self.stable} generations.")
        return

    def evaluate_order(self, order):
        """
        Evaluate one candidate ordering and return the results.
        :param order: the candidate ordering.
        :return: the solution and whether the set of local optimal solution gets updated.
        """
        solution, var = self.solver(order, self.mask)
        update = self.add_opt(solution, var, order)
        return solution, update

    def refine_solution(self, num_refine):
        """
        Refine a set of local optimal solutions (e.g., try to improve the solution by evolving until termination).
        :param num_refine: number of local optimal solutions to refine.
        """
        return

    def crossover(self):
        """Perform crossover to generate a new group of orderings from the survivors."""
        # Exchange the order at some hops selected probabilistically for each pair of parents.
        # Select a subset (for enumeration) from the domain of flow orderings when the size of domain is small.
        if self.enum:
            max_num = min(len(self.order_set), (self.generation + 1) * self.opts.group_size)
            self.group = self.order_set[self.generation * self.opts.group_size:max_num]
            return
        # Generate a set of modified flow orderings for each pair of parent orderings.
        children = list()
        num_gen = max(self.opts.group_size // len(self.survivor), 1)
        parent1, parent2 = self.survivor[::2], self.survivor[1::2]
        for p1, p2 in zip(parent1, parent2):
            for _ in range(num_gen):
                c1, c2 = self.cross_parent(p1, p2)
                children.append(c1)
                children.append(c2)
        # Append the last singular survivor (if any).
        if len(parent1) > len(parent2):
            children.append(parent1[-1])
        self.group = np.array(children)
        return

    def cross_parent(self, p1, p2):
        """
        Perform crossover on one pair of parents to generate a pair of children.
        :param p1: the first parent.
        :param p2: the second parent.
        :return: the pair of children.
        """
        exchange_link = np.random.rand(self.num_link) < self.opts.cross_rate
        exchange_mask = np.zeros_like(self.mask, dtype=int)
        exchange_mask[self.cut[:-1][exchange_link]] += 1
        exchange_mask[self.cut[1:-1][exchange_link[:-1]]] -= 1
        exchange_mask = np.cumsum(exchange_mask).astype(bool)
        c1 = np.where(exchange_mask, p1, p2)
        c2 = np.where(exchange_mask, p2, p1)
        return c1, c2

    def mutate(self):
        """Perform mutation to modify the group of orderings."""
        return

    def check_opts(self):
        """Sanity check on the input options."""
        # TODO: Make sure in random sample mode the total number of orderings is no smaller than the group size.
        return

    def total_num(self):
        """Compute the logarithm of the total number of orderings in the domain space."""
        return

    def set_bounds(self):
        """
        Set the maximum number of (unique) orderings the algorithm will explore.
        Set whether the algorithm needs to enumerate and explore all feasible orderings.
        """
        # Compute some thresholds using Newton's method.
        thresh1 = self.get_thresh(1, self.opts.log_base)
        thresh2 = self.get_thresh(2, self.opts.log_base)
        # Set upper bound and enumeration mode according to the thresholds computed.
        log_total = self.total / np.log(self.opts.log_base) + 1
        ub = log_total if log_total > thresh1 else np.rint(np.exp(self.total))
        self.ub = int(ub)
        self.enum = log_total <= thresh2 / 2 + 0.5
        self.opts.local_size = min(self.opts.local_size, self.ub)
        return

    def set_mask(self):
        """Set mask that can segment flow ordering onto each hop."""
        return

    def construct_set(self):
        """Construct a set of all feasible orderings through enumeration when the domain is not too large."""
        return

    def get_optimal(self):
        """Return the optimal solution once the algorithm terminates."""
        best_idx = np.argmin(self.opt_solution)
        best_solution = self.opt_solution[best_idx]
        best_var = self.opt_var[best_idx]
        best_order = self.opt_order[best_idx]
        return best_solution, best_var, best_order

    def add_opt(self, solution, var, order):
        """
        Helper function to update current best solutions.
        :param solution: value of a newly found solution.
        :param var: the variables associated with the solution.
        :param order: the ordering that specifies the solution.
        :return: whether the set of local optimal solutions gets updated.
        """
        max_solution, max_idx = np.amax(self.opt_solution), np.argmax(self.opt_solution)
        min_solution = np.amin(self.opt_solution)
        if solution < max_solution:
            self.opt_solution[max_idx] = solution
            self.opt_var[max_idx] = var
            self.opt_order[max_idx] = order
        update = solution < min_solution - self.opts.err_tolerance
        return update

    def get_thresh(self, scale, base):
        """
        Helper function to compute the threshold value x for the equation log_a(x) = (x - 1) / s,
        :param scale: the scale (s) in the above equation.
        :param base: the base (a) of the log function in the above equation.
        :return: the threshold value.
        """
        x_init = 100
        thresh = 1
        if np.log(base) < scale:
            while abs(thresh - 1) <= self.opts.err_tolerance:
                thresh = newton_method(lambda x: np.log(x) / np.log(base) - (x - 1) / scale,
                                       lambda x: 1 / (x * np.log(base)) - 1 / scale, x_init, self.opts.err_tolerance)
                x_init *= x_init
        return thresh
