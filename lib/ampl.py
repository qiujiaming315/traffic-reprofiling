import numpy as np
import os

"""Functions to translate the input (network, flow order) into a non-linear program instance formulated as AMPL files."""
# TODO: Delete this file after finishing the implementation.


def num_digit(value):
    """Get the number of digits in an integer."""
    return len(str(value))


def formulate_command(path):
    """Generate the AMPL command file (with .run extension)."""
    lines = list()
    lines.append("solve;")
    lines.append("display Total_BW;")
    lines.append("display Link_BW;")
    lines.append("display InNet_DDL;")
    lines.append("display PerHop_DDL;")
    with open(f"{path}minband.run", "w") as f:
        for line in lines:
            f.write(line + '\n')
    return


def formulate_data(route, flow_profile):
    """Generate the AMPL data file (with .dat extension)."""
    route = route.astype(int)
    num_flow, num_link = route.shape
    heading, declare, route_param, flow_param = list(), list(), list(), list()
    # File heading.
    heading.append("data;")
    # Set declaration.
    declare.append("#Declare the flow and link indices as set.")
    nf_space, nl_space = num_digit(num_flow) + 1, num_digit(num_link) + 1
    val_space = [num_digit(x) for x in flow_profile.flatten()]
    val_space = np.amax(val_space) + 1
    flow_set, link_set = "set FLOW := ", "set LINK := "
    for i in range(num_flow):
        flow_set += f"{i:<{nf_space}}"
    for i in range(num_link):
        link_set += f"{i:<{nl_space}}"
    flow_set += ";"
    link_set += ";"
    declare.append(flow_set)
    declare.append(link_set)
    # Specify parameters.
    # Set the route parameter.
    route_param.append("#Route of each flow.")
    route_param.append("param route :")
    route_heading = ' ' * nf_space
    route_heading += '\t'
    for i in range(num_link):
        route_heading += f"{i:<{nl_space}}"
    route_heading += ":="
    route_param.append(route_heading)
    for i in range(num_flow):
        flow_line = f"{i:<{nf_space}}\t"
        for j in range(num_link):
            flow_line += f"{route[i, j]:<{nl_space}}"
        if i == num_flow - 1:
            flow_line += ";"
        route_param.append(flow_line)
    # Set the flow profile parameters.
    flow_param.append("#Flow profiles.")
    fp_names = ["rate", "burst", "e2e_ddl"]
    max_cspace = max(len(fp_names[2]) + 1, val_space)
    max_rspace = max(6, nf_space)
    flow_heading = "param:"
    flow_heading = f"{flow_heading:<{max_rspace}}\t"
    for name in fp_names:
        flow_heading += f"{name:<{max_cspace}}"
    flow_heading += ":="
    flow_param.append(flow_heading)
    for i in range(num_flow):
        flow_line = f"{i:<{max_rspace}}\t"
        for j in range(3):
            flow_line += f"{flow_profile[i, j]:<{max_cspace}}"
        if i == num_flow - 1:
            flow_line += ";"
        flow_param.append(flow_line)
    return heading, declare, route_param, flow_param


def formulate_model(path, num_link):
    """Generate the AMPL model file (with .mod extension)."""
    objective = list()
    objective.append("#Declare the sets, parameters, and variables.")
    objective.append("set FLOW;")
    objective.append("set LINK;")
    for i in range(num_link):
        objective.append(f"set ORDER{i} ordered;")
    objective.append("")
    objective.append("param route {FLOW, LINK} binary;")
    objective.append("param order {LINK};")
    objective.append("param rate {FLOW} >= 0;")
    objective.append("param burst {FLOW} >= 0;")
    objective.append("param e2e_ddl {FLOW} > 0;")
    objective.append("")
    objective.append("var Link_BW {LINK} >= 0;")
    objective.append("var InNet_DDL {FLOW} >= 0;")
    objective.append("var PerHop_DDL {FLOW, LINK} >= 0;")
    objective.append("")
    objective.append("#Optimization objective.")
    objective.append("minimize Total_BW: sum {j in LINK} Link_BW[j];")
    deadline_constraints = list()
    deadline_constraints.append("#In-network deadline constraints.")
    deadline_constraints.append("subj to MaxDDL {i in FLOW}: InNet_DDL[i] <= e2e_ddl[i];")
    deadline_constraints.append(
        "subj to MinDDL {i in FLOW}: InNet_DDL[i] >= e2e_ddl[i] - (burst[i] / rate[i]);")
    deadline_constraints.append("#Deadline summation equality constraint.")
    deadline_constraints.append(
        "subj to DDLSum {i in FLOW}: sum {j in LINK} route[i, j] * PerHop_DDL[i, j] = InNet_DDL[i];")
    deadline_constraints.append("#Deadline order constraints.")
    for i in range(num_link):
        deadline_constraints.append(
            f"subj to DDLOrder{i} {{o in ORDER{i}: ord(o) > 1}}: PerHop_DDL[o, {i}] >= PerHop_DDL[prev(o), {i}];")
    # deadline_constraints.append(
    #     "subj to DDLOrder {j in LINK, o in order[j]: ord(o) > 1}: " + \
    #     "PerHop_DDL[o, j] >= PerHop_DDL[prev(o), j];")
    bandwidth_constraints = list()
    bandwidth_constraints.append("#Bandwidth constraints.")
    bandwidth_constraints.append(
        "subj to LeastBW {j in LINK}: Link_BW[j] >= sum {i in FLOW} route[i, j] * rate[i];")
    for i in range(num_link):
        bandwidth_constraints.append(
            f"subj to MinBW{i} {{o in ORDER{i}}}: Link_BW[{i}] >= sum {{t in ORDER{i}: ord(t) <= ord(o)}} " + \
            f"(burst[t] - (rate[t] * (e2e_ddl[t] - InNet_DDL[t])) + " + \
            f"(rate[t] * (PerHop_DDL[o, {i}] - PerHop_DDL[t, {i}]))) / PerHop_DDL[o, {i}];")
    # bandwidth_constraints.append(
    #     "subj to MinBW {j in LINK, o in order[j]}: Link_BW[j] >= sum {t in order[j]: ord(t) <= ord(o)} " + \
    #     "(burst[t] + (rate[t] * (PerHop_DDL[o, j] - PerHop_DDL[t, j]))) / PerHop_DDL[o, j];")
    # Write to the model file.
    with open(f"{path}minband.mod", "w") as f:
        for segment in [objective, deadline_constraints, bandwidth_constraints]:
            for line in segment:
                f.write(line + '\n')
            f.write('\n')
    return


def formulate(route, flow_profile, path):
    """
    Formulate the input as a non-linear program instance described with AMPL files.

    route, flow_profile: the route and profile of flows.
    path: directory to save generated AMPL files.

    Return a translator that takes in flow order at every link and finish the formulation.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, "")
    num_flow, num_link = route.shape
    formulate_command(path)
    formulate_model(path, num_link)
    heading, declare, route_param, flow_param = formulate_data(route, flow_profile)

    def set_order(order, mask, index=0):
        # # Set the flow deadline order parameter.
        # nl_space = num_digit(num_link) + 1
        # order_param = list()
        # order_param.append("#Deadline order of flow at each link.")
        # order_param.append("param order :=")
        # for i in range(num_link):
        #     link_order = order[:, i][route[:, i][order[:, i]]]
        #     order_line = f"{i:<{nl_space}}\t" + "{" + f"{link_order[0]}"
        #     for idx in link_order[1:]:
        #         order_line += f", {idx}"
        #     order_line += "} ordered"
        #     if i == num_link - 1:
        #         order_line += ";"
        #     order_param.append(order_line)

        # Complete the data file.
        # Set the flow deadline order.
        order_set = list()
        order_set.append("#Declare the flow deadline order as ordered sets.")
        for link_idx in range(num_link):
            link_order = order[mask == link_idx]
            order_line = f"set ORDER{link_idx} :="
            for flow_idx in link_order:
                order_line += f" {flow_idx}"
            order_line += ";"
            order_set.append(order_line)
        # for i in range(num_link):
        #     link_order = order[:, i][route[:, i][order[:, i]]]
        #     order_line = f"set ORDER{i} :="
        #     for idx in link_order:
        #         order_line += f" {idx}"
        #     order_line += ";"
        #     order_set.append(order_line)
        # Write to the data file.
        with open(f"{path}minband{index}.dat", "w") as f:
            for segment in [heading, declare, order_set, route_param, flow_param]:
                for line in segment:
                    f.write(line + '\n')
                f.write('\n')
        return

    return set_order
