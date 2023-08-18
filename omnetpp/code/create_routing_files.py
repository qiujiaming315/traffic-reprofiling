import numpy as np
import argparse
import os
from pathlib import Path

"""Generate ned and ini files for OMNeT++ to compute routing."""


def compose_ned(net_data, option, size='s'):
    """
    Compose the initialization file for OMNeT++.
    :param net_data: network data containing node and link information.
    :param option: option parameters.
    :param size: icon size, can be 's' or 'vs'.
    """
    # Import the modules and set the routing table recorder and simulation visualizer.
    ned_content = list()
    ned_content.append(f"package {option.package_name};")
    ned_content.append("")
    ned_content.append("import inet.visualizer.common.IntegratedVisualizer;")
    ned_content.append("import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;")
    ned_content.append("import inet.networklayer.ipv4.RoutingTableRecorder;")
    ned_content.append("import inet.common.scenario.ScenarioManager;")
    ned_content.append("import inet.node.inet.Router;")
    ned_content.append("import inet.node.inet.StandardHost;")
    ned_content.append("")
    ned_content.append(f"network {option.ned_name}")
    ned_content.append("{")
    ned_content.append("    @display(\"bgb=763,478\");")
    ned_content.append("    submodules:")
    ned_content.append("        rtr: RoutingTableRecorder {")
    ned_content.append("            @display(\"p=660,120\");")
    ned_content.append("        }")
    ned_content.append("        visualizer: IntegratedVisualizer {")
    ned_content.append("            @display(\"p=660,216\");")
    ned_content.append("        }")
    # Configure each node (standard host or router) in the network.
    nodes, node_names, links = net_data["nodes"], net_data["node_names"], net_data["links"]
    interface_count = np.zeros((len(nodes),), dtype=int)
    for link in links:
        src, dest = link[0], link[1]
        interface_count[src] += 1
        interface_count[dest] += 1
    for node, name, count in zip(nodes, node_names, interface_count):
        node_type, node_coordx, node_coody = node[0], node[1], node[2]
        type_name = "StandardHost" if node_type == 0 else "Router"
        ned_content.append(f"        {name}: {type_name}" + " {")
        ned_content.append("            parameters:")
        if node_type == 1:
            ned_content.append("                hasRip = true;")
        ned_content.append(f"                @display(\"p={node_coordx},{node_coody};is={size}\");")
        ned_content.append("            gates:")
        ned_content.append(f"                pppg[{count}];")
        ned_content.append("        }")
    # Set the IPv4 configurator.
    ned_content.append("        configurator: Ipv4NetworkConfigurator {")
    ned_content.append("            @display(\"p=660,309\");")
    ned_content.append("            config = xml(\"<config>\"")
    configurator_prefix = " " * 28
    core_net, access_net, gate = list(), list(), list()
    core_count, access_count = 0, 0
    for link in links:
        src, dest = link[0], link[1]
        src_type, dest_type = nodes[src, 0], nodes[dest, 0]
        link_type = src_type + dest_type
        assert link_type != 0, "Hosts should not directly connect to each other."
        if link_type == 1:
            host_name = node_names[src] if src_type == 0 else node_names[dest]
            router_name = node_names[src] if src_type == 1 else node_names[dest]
            access_net.append(configurator_prefix + f"+\"<interface among='{host_name} {router_name}' "
                                                    f"address='192.168.{access_count}.x' netmask='255.255.255.x'/>\"")
            gate.append(configurator_prefix + f"+\"<route hosts='{host_name}' destination='*' "
                                              f"gateway='{router_name}'/>\"")
            access_count += 1
        else:
            core_net.append(configurator_prefix + f"+\"<interface among='{node_names[src]} {node_names[dest]}' "
                                                  f"address='10.{core_count}.0.x' netmask='255.255.255.x' "
                                                  f"add-static-route='off'/>\"")
            core_count += 1
    assert core_count <= 256 and access_count <= 256, "Cannot assign enough IP addresses for all the subnets."
    ned_content.extend(core_net)
    ned_content.extend(access_net)
    ned_content.extend(gate)
    ned_content.append("                        +\"</config>\");")
    ned_content.append("            addStaticRoutes = false;")
    ned_content.append("        }")
    # Set the scenario manager.
    ned_content.append("        scenarioManager: ScenarioManager {")
    ned_content.append("            @display(\"100=100;p=660,397\");")
    ned_content.append("        }")
    # Set each link in the network.
    ned_content.append("    connections:")
    interface_idx = np.zeros((len(nodes),), dtype=int)
    for link in links:
        src, dest = link[0], link[1]
        # Set bandwidth to an infinite value (just for routing).
        inf_bw = "100000000000000000000000000000000000000000000000000Gbps"
        ned_content.append(f"        {node_names[src]}.pppg[{interface_idx[src]}] <--> {{datarate = {inf_bw};}} "
                           f"<--> {node_names[dest]}.pppg[{interface_idx[dest]}];")
        interface_idx[src] += 1
        interface_idx[dest] += 1
    ned_content.append("}")
    # Write the composed content to the specified file.
    with open(os.path.join(option.out, option.ned_name + ".ned"), "w") as f:
        f.write("\n".join(ned_content))
    return


def compose_ini(option):
    """
    Compose the initialization file for OMNeT++.
    :param option: option parameters.
    """
    ini_content = list()
    ini_content.append("[General]")
    ini_content.append(f"network = {option.ned_name}")
    ini_content.append(f"description = \"{option.ned_name}\"")
    ini_content.append("sim-time-limit = 30s")
    ini_content.append("")
    ini_content.append("")
    ini_content.append("#----------------rip interval configuration----------------")
    ini_content.append("**.R*.rip.updateInterval = 1000s")
    ini_content.append("**.R*.rip.routeExpiryTime = 2000s")
    ini_content.append("**.R*.rip.routePurgeTime = 1800s")
    with open(os.path.join(option.out, option.ini_name + ".ini"), "w") as f:
        f.write("\n".join(ini_content))
    return


def main(opts):
    # Load the network data.
    network_data = np.load(opts.net)
    # Create the directory to save the simulation files.
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    # Compose the ned file.
    compose_ned(network_data, opts, size='vs')
    # Compose the ini file.
    compose_ini(opts)
    return


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('net', help="Path to the input npz file describing network nodes and links.")
    args.add_argument('out', help="Directory to save results.")
    args.add_argument('--ned_name', type=str, default="test", help="Name of the ned file.")
    args.add_argument('--ini_name', type=str, default="test", help="Name of the ini file.")
    args.add_argument('--package_name', type=str, default="test", help="Package name of the ned file.")
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
