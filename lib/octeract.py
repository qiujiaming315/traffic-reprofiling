import numpy as np
import octeract


def set_basic(model, path_matrix, flow_profile, objective, weight):
    """
    Set the objective function and basic constraints for the NLP model.
    :param model: the template (non-linear program object) to formulate non-linear programs.
    :param path_matrix: network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: the number of basic (independent of flow orderings) constraints for this network.
    """
    # Set the model variables.
    num_flow, num_link = path_matrix.shape
    for flow_idx in range(num_flow):
        model.add_variable(f"D{flow_idx}", 0)
    for link_idx in range(num_link):
        model.add_variable(f"C{link_idx}", 0)
    for flow_idx, flow_route in enumerate(path_matrix):
        for link_idx, link_flow in enumerate(flow_route):
            if link_flow:
                model.add_variable(f"T{flow_idx}_{link_idx}", 0)
    # Set the objective function.
    if objective == 0:
        obj_str = [f"C{link_idx}" for link_idx in range(num_link)]
        obj_str = " + ".join(obj_str)
    elif objective == 1:
        obj_str = [f"{wt} * C{link_idx}" for link_idx, wt in enumerate(weight)]
        obj_str = " + ".join(obj_str)
    else:
        model.add_variable(f"C_max", 0)
        for link_idx in range(num_link):
            model.add_constraint(f"C_max >= C{link_idx}")
        obj_str = "C_max"
    model.set_objective(obj_str)
    # Set the deadline constraints.
    total_ddl, sd_ub = flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0]
    for flow_idx, (ddl, ub) in enumerate(zip(total_ddl, sd_ub)):
        model.add_constraint(f"D{flow_idx} <= {ddl}")
        model.add_constraint(f"D{flow_idx} <= {ub}")
        links = np.arange(num_link)[path_matrix[flow_idx]]
        ddl_str = [f"T{flow_idx}_{link_idx}" for link_idx in links]
        ddl_str = " + ".join(ddl_str)
        ddl_str += f" <= {ddl} - D{flow_idx}"
        model.add_constraint(ddl_str)
    # Set the bandwidth constraints.
    for link_idx in range(num_link):
        link_lb = np.sum(flow_profile[:, 0][path_matrix[:, link_idx]])
        model.add_constraint(f"C{link_idx} >= {link_lb}")
    return model.num_constraints()


def formulate(path_matrix, flow_profile, objective, weight):
    """
    Formulate the input as a non-linear program instance and solve it with octeract solver.
    :param path_matrix: network routes.
    :param flow_profile: the flow profile.
    :param objective: the objective function.
    :param weight: the bandwidth weight profile.
    :return: a function that takes as input flow orderings at every hop, generates the corresponding non-linear program
             instance, and returns the solution of this instance.
    """
    _, num_link = path_matrix.shape
    model = octeract.Model()
    num_constraint = set_basic(model, path_matrix, flow_profile, objective, weight)

    def two_slope_solver(order, mask, local=True):
        # Remove the order-dependent constraints set by previous trails.
        constraints = model.get_constraint_names()
        for cn in constraints[num_constraint:]:
            model.remove_constraint(cn)
        # Set the order-dependent constraints.
        for link_idx in range(num_link):
            link_order = order[mask == link_idx]
            base_str, base_r, base_ddl = "( 0", 0, "0"
            rate_list, flow_set = list(), set()
            for flow_idx in link_order:
                fr, fb, fd = flow_profile[flow_idx]
                if flow_idx in flow_set:
                    new_ddl = f"( T{flow_idx}_{link_idx} + D{flow_idx} )"
                    base_str += f" + ( {' + '.join(rate_list)} + {base_r} ) * ( {new_ddl} - {base_ddl} )"
                    model.add_constraint(f"C{link_idx} >= " + base_str + f" ) / {new_ddl}")
                    model.add_constraint(f"{new_ddl} >= {base_ddl}")
                    rate_list.remove(f"{fb} / D{flow_idx}")
                    base_r += fr
                    base_ddl = new_ddl
                else:
                    flow_set.add(flow_idx)
                    new_ddl = f"T{flow_idx}_{link_idx}"
                    rate_str = " + ".join(rate_list) if len(rate_list) > 0 else "0"
                    base_str += f" + ( {rate_str} + {base_r} ) * ( {new_ddl} - {base_ddl} )"
                    model.add_constraint(f"{new_ddl} >= {base_ddl}")
                    rate_list.append(f"{fb} / D{flow_idx}")
                    base_ddl = new_ddl
        # Solve the model.
        if local:
            model.local_solve()
        else:
            model.global_solve()
        return model.get_solution_objective_value(), model.get_solution_vector()

    return two_slope_solver
