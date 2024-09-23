import numpy as np
import pyomo.environ as pyo
import re


def set_basic(model, path_matrix, flow_profile, objective, weight, scheduler="fifo"):
    """
    Set the objective function and basic constraints for the NLP model.
    :param model: the template (non-linear program object) to formulate non-linear programs.
    :param path_matrix: network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :param scheduler: the schedulers applied at each hop in the network.
    """
    # Set the model variables.
    num_flow, num_link = path_matrix.shape
    for flow_idx in range(num_flow):
        setattr(model, f"D{flow_idx}", pyo.Var(initialize=0.0, within=pyo.NonNegativeReals))
    for link_idx in range(num_link):
        setattr(model, f"C{link_idx}", pyo.Var(initialize=0.0, within=pyo.NonNegativeReals))
    # Add local deadline variables according to the schedulers being used.
    if scheduler == "fifo":
        for link_idx in range(num_link):
            setattr(model, f"T{link_idx}", pyo.Var(initialize=0.0, within=pyo.NonNegativeReals))
    elif scheduler == "sced":
        for flow_idx, flow_route in enumerate(path_matrix):
            for link_idx, link_flow in enumerate(flow_route):
                if link_flow:
                    setattr(model, f"T{flow_idx}_{link_idx}", pyo.Var(initialize=0.0, within=pyo.NonNegativeReals))
    # Set the objective function.
    if objective == 0:
        obj_expr = sum(getattr(model, f"C{link_idx}") for link_idx in range(num_link))
    elif objective == 1:
        obj_expr = sum(wt * getattr(model, f"C{link_idx}") for link_idx, wt in enumerate(weight))
    else:
        model.C_max = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
        for link_idx in range(num_link):
            setattr(model, f"C_max_ct{link_idx}", pyo.Constraint(
                expr=(model.C_max >= getattr(model, f"C{link_idx}"))
            ))
        obj_expr = model.C_max
    model.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    # Set the deadline constraints.
    total_ddl, sd_ub = flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0]
    for flow_idx, (ddl, ub) in enumerate(zip(total_ddl, sd_ub)):
        setattr(model, f"D{flow_idx}_ct0", pyo.Constraint(
            expr=(getattr(model, f"D{flow_idx}") <= ddl)
        ))
        setattr(model, f"D{flow_idx}_ct1", pyo.Constraint(
            expr=(getattr(model, f"D{flow_idx}") <= ub)
        ))
        links = np.arange(num_link)[path_matrix[flow_idx]]
        if scheduler == "fifo":
            ddl_expr = sum(getattr(model, f"T{link_idx}") for link_idx in links)
        elif scheduler == "sced":
            ddl_expr = sum(getattr(model, f"T{flow_idx}_{link_idx}") for link_idx in links)
        setattr(model, f"ddl_ct{flow_idx}", pyo.Constraint(expr=(ddl_expr <= ddl - getattr(model, f"D{flow_idx}"))))
    # Set the stability constraints.
    for link_idx in range(num_link):
        link_lb = np.sum(flow_profile[:, 0][path_matrix[:, link_idx]])
        setattr(model, f"stable{link_idx}", pyo.Constraint(expr=(getattr(model, f"C{link_idx}") >= link_lb)))
    return


def delete_component(model, prefix):
    """Helper function to delete constraints with a specified prefix."""
    list_del = [vr for vr in vars(model) if re.match(prefix, vr)]

    for kk in list_del:
        model.del_component(kk)


def formulate_fifo(path_matrix, flow_profile, objective, weight):
    """
    Formulate and solve a non-linear program instance (for FIFO schedulers).
    :param path_matrix: network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: a function that takes as input flow orderings at every hop, generates the corresponding non-linear program
             instance, and returns the solution of this instance.
    """
    num_flow, num_link = path_matrix.shape
    model = pyo.ConcreteModel()
    solver = pyo.SolverFactory("ipopt")
    set_basic(model, path_matrix, flow_profile, objective, weight, "fifo")

    def two_slope_solver(order):
        # Delete the order-dependent constraints set earlier.
        delete_component(model, r"C[0-9]+_ct[0-9]+")
        delete_component(model, r"order_ct[0-9]+")
        # Set the new constraints based on the given order.
        for ct_idx, (flow_low, flow_high) in enumerate(zip(order[:-1], order[1:])):
            setattr(model, f"order_ct{ct_idx}",
                    pyo.Constraint(
                        expr=(getattr(model, f"D{flow_high}") >= getattr(model, f"D{flow_low}"))))
        for link_idx in range(num_link):
            link_mask = path_matrix[:, link_idx]
            link_mask = link_mask[order]
            link_order = order[link_mask]
            base_expr, base_r, base_ddl = 0, 0, 0
            for ct_idx, flow_idx in enumerate(link_order):
                fr, fb, fd = flow_profile[flow_idx]
                new_ddl = getattr(model, f"D{flow_idx}")
                base_expr += (sum(
                    flow_profile[f_idx, 1] / getattr(model, f"D{f_idx}") for f_idx in link_order[ct_idx:]) + base_r) * (
                                     new_ddl - base_ddl)
                setattr(model, f"C{link_idx}_ct{ct_idx}", pyo.Constraint(
                    expr=(getattr(model, f"C{link_idx}") >= base_expr / (getattr(model, f"T{link_idx}") + new_ddl))))
                base_r += fr
                base_ddl = new_ddl
        # Solve the model.
        try:
            solver.solve(model)
        except ValueError:
            print("NLP instance is infeasible.")
            return np.inf, None
        # Retrieve the solution variables.
        variables = dict()
        for flow_idx in range(num_flow):
            variables[f"D{flow_idx}"] = getattr(model, f"D{flow_idx}")()
        for link_idx in range(num_link):
            variables[f"C{link_idx}"] = getattr(model, f"C{link_idx}")()
            variables[f"T{link_idx}"] = getattr(model, f"T{link_idx}")()
        return model.obj(), variables

    return two_slope_solver


def formulate_sced(path_matrix, flow_profile, objective, weight):
    """
    Formulate and solve a non-linear program instance (for SCED schedulers).
    :param path_matrix: network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: a function that takes as input flow orderings at every hop, generates the corresponding non-linear program
             instance, and returns the solution of this instance.
    """
    num_flow, num_link = path_matrix.shape
    model = pyo.ConcreteModel()
    solver = pyo.SolverFactory("ipopt")
    set_basic(model, path_matrix, flow_profile, objective, weight, "sced")

    def two_slope_solver(order, mask):
        # Delete the order-dependent constraints set earlier.
        delete_component(model, r"C[0-9]+_ct[0-9]+")
        delete_component(model, r"order[0-9]+_ct[0-9]+")
        # Set the new constraints based on the given order.
        for link_idx in range(num_link):
            link_order = order[mask == link_idx]
            base_expr, base_r, base_ddl = 0, 0, 0
            rate_list, flow_set = list(), set()
            for ct_idx, flow_idx in enumerate(link_order):
                if flow_idx in flow_set:
                    new_ddl = getattr(model, f"T{flow_idx}_{link_idx}") + getattr(model, f"D{flow_idx}")
                    base_expr += (sum(
                        flow_profile[f_idx, 1] / getattr(model, f"D{f_idx}") for f_idx in rate_list) + base_r) * (
                                         new_ddl - base_ddl)
                    setattr(model, f"C{link_idx}_ct{ct_idx}", pyo.Constraint(
                        expr=(getattr(model, f"C{link_idx}") >= base_expr / new_ddl)))
                    setattr(model, f"order{link_idx}_ct{ct_idx}", pyo.Constraint(
                        expr=(new_ddl >= base_ddl)))
                    rate_list.remove(flow_idx)
                    base_r += flow_profile[flow_idx, 0]
                    base_ddl = new_ddl
                else:
                    flow_set.add(flow_idx)
                    new_ddl = getattr(model, f"T{flow_idx}_{link_idx}")
                    base_expr += (sum(
                        flow_profile[f_idx, 1] / getattr(model, f"D{f_idx}") for f_idx in rate_list) + base_r) * (
                                         new_ddl - base_ddl)
                    setattr(model, f"order{link_idx}_ct{ct_idx}", pyo.Constraint(
                        expr=(new_ddl >= base_ddl)))
                    rate_list.append(flow_idx)
                    base_ddl = new_ddl
        # Solve the model.
        try:
            solver.solve(model)
        except ValueError:
            print("NLP instance is infeasible.")
            return np.inf, None
        # Retrieve the solution variables.
        variables = dict()
        for flow_idx in range(num_flow):
            variables[f"D{flow_idx}"] = getattr(model, f"D{flow_idx}")()
        for link_idx in range(num_link):
            variables[f"C{link_idx}"] = getattr(model, f"C{link_idx}")()
        for flow_idx, flow_route in enumerate(path_matrix):
            for link_idx, flow_link in enumerate(flow_route):
                if flow_link:
                    variables[f"T{flow_idx}_{link_idx}"] = getattr(model, f"T{flow_idx}_{link_idx}")()
        return model.obj(), variables

    return two_slope_solver
