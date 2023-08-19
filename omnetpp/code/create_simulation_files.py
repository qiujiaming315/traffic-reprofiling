import numpy as np
import argparse
import os
from pathlib import Path

"""Generate the configuration files for OMNeT++ to run simulations."""


def flow_count(net_data, flows, save_dir):
    """
    Collect data that describes network topology and write flow filter files.
    :param net_data: network data containing node and link information.
    :param flows: the route of each flow.
    :param save_dir: directory to save xml files that filter packets from different flows.
    :return: the list of flow(s) going through each link interface.
    """
    # Determine which link interface goes to which node.
    nodes, node_names, links = net_data["nodes"], net_data["node_names"], net_data["links"]
    interface_idx_list = [list() for _ in range(len(nodes))]
    for link in links:
        src, dest = link[0], link[1]
        interface_idx_list[src].append(dest)
        interface_idx_list[dest].append(src)
    # Count the number of flows going through each interface.
    interface_flow_list = list()
    for interfaces in interface_idx_list:
        interface_flow_list.append([list() for _ in range(len(interfaces))])
    for flow_idx, route in enumerate(flows):
        flow_nodes = np.where(route)[0]
        flow_nodes = flow_nodes[np.argsort(route[flow_nodes])]
        for start, end in zip(flow_nodes[:-1], flow_nodes[1:]):
            interface_idx = interface_idx_list[start].index(end)
            interface_flow_list[start][interface_idx].append(flow_idx)
    # Create the filter.xml files.
    for node_idx in range(len(interface_flow_list)):
        for interface_idx in range(len(interface_flow_list[node_idx])):
            filter_content = ["<filters>"]
            for gate_idx in range(len(interface_flow_list[node_idx][interface_idx])):
                flow_idx = interface_flow_list[node_idx][interface_idx][gate_idx]
                flow_src = np.where(flows[flow_idx] == 1)[0][0]
                flow_dest = np.argmax(flows[flow_idx])
                filter_content.append(
                    f"  <filter srcAddress=\"{node_names[flow_src]}\" destAddress=\"{node_names[flow_dest]}\" "
                    f"protocol=\"udp\" destPort=\"{flow_idx + 1000}\" gate=\"{gate_idx}\"/>")
            filter_content.append("</filters>")
            with open(os.path.join(save_dir, f"filters_{node_names[node_idx]}{interface_idx}.xml"), "w") as file:
                file.write("\n".join(filter_content))
    return interface_flow_list


def compose_ini(interface_flow_list, net_data, flow_route, flow_profile, reprofiling_data, link_bandwidth,
                traffic_pattern, option):
    """
    Compose the initialization and scenario manager file for OMNeT++.
    :param interface_flow_list: the list of flow(s) going through each link interface.
    :param net_data: network data containing node and link information.
    :param flow_route: the route of each flow.
    :param flow_profile: the flow profiles.
    :param reprofiling_data: the reprofiling parameters of each flow.
    :param link_bandwidth: bandwidth of each link. Each bidirectional link is decoupled into two unidirectional links.
    :param traffic_pattern: the traffic arrival pattern of each flow.
    :param option: option parameters.
    """
    # Set the basics of the configuration files.
    ini_content = []
    scenario_content = ["<scenario>"]
    ini_content.append(f"[{option.ned_name}]")
    ini_content.append(f"network = {option.ned_name}")
    ini_content.append(f"description = \"{option.ned_name}\"")
    ini_content.append(f"sim-time-limit = {option.simulation_time}s\n\n")
    # Set the routing protocol not to update routes during the simulation.
    ini_content.append("#----------------rip interval configuration----------------")
    ini_content.append("**.R*.rip.updateInterval = 1000s")
    ini_content.append("**.R*.rip.routeExpiryTime = 2000s")
    ini_content.append("**.R*.rip.routePurgeTime = 1800s")
    # Retrieve data.
    nodes, node_names, links = net_data["nodes"], net_data["node_names"], net_data["links"]
    reprofiling_delay = reprofiling_data['reprofiling_delay']
    app_idx_count = [0 for _ in range(len(nodes))]
    # Configure the applications.
    ini_content.append("#----------------udp application configuration----------------")
    for i in range(len(flow_route)):
        flow_src = np.where(flow_route[i] == 1)[0][0]
        src_name = node_names[flow_src]
        app_num_src = app_idx_count[flow_src]
        flow_dest = np.argmax(flow_route[i])
        dest_name = node_names[flow_dest]
        app_num_dest = app_idx_count[flow_dest]
        # Data sender configurations.
        ini_content.append(f"#----------------flow {i + 1}----------------")
        ini_content.append(f"#----{src_name}----")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].typename = \"UdpBasicBurst\"")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].destAddresses = \"{dest_name}\"")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].destPort = {i + 1000}")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].chooseDestAddrMode = \"perBurst\"")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].messageLength = {option.packet_size - 28}B")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].sendInterval = 1ps")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].startTime = {traffic_pattern[i][0]}s")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].stopTime = {traffic_pattern[i][1]}s")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].burstDuration = {traffic_pattern[i][2]}s")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].sleepDuration = {traffic_pattern[i][3]}s")
        ini_content.append(f"**.{src_name}.app[{app_num_src}].delayLimit = 0s")
        # Data receiver configurations.
        app_idx_count[flow_src] += 1
        ini_content.append(f"#----{dest_name}----")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].typename = \"UdpBasicBurst\"")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].destAddresses = \"\"")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].localPort = {i + 1000}")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].destPort = 0")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].chooseDestAddrMode = \"once\"")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].messageLength = 0B")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].sendInterval = 0s")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].sleepDuration = 0s")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].burstDuration = 0s")
        ini_content.append(f"**.{dest_name}.app[{app_num_dest}].delayLimit = 0s")
        app_idx_count[flow_dest] += 1
        # Manage the traffic arrival pattern of each flow.
        cycle_time = traffic_pattern[i][2] + traffic_pattern[i][3]
        cycle_num = (traffic_pattern[i][1] - traffic_pattern[i][0]) / cycle_time
        cycle_num = max(1, int(cycle_num))
        for j in range(cycle_num):
            reset_time = ((traffic_pattern[i][0] + j * cycle_time) * 1e12 + flow_profile[i][
                1] / option.packet_size) / 1e12
            send_interval = round(1e6 / flow_profile[i][0] * option.packet_size, 3)
            scenario_content.append(
                f"    <at t=\"{reset_time}\">")
            scenario_content.append(
                f"        <set-param module=\"{src_name}.app[{app_num_src}]\" "
                f"par=\"sendInterval\" value=\"{send_interval}us\"/>")
            scenario_content.append("    </at>")
    # Set the application number of each end-host.
    for i in range(len(nodes)):
        if nodes[i][0] == 0:
            ini_content.append(f"**.{node_names[i]}.numApps = {app_idx_count[i]}")
    # Configure the reprofilers at each link layer interface.
    ini_content.append("#----------------Traffic Conditioner within PPP interface----------------")
    for i in range(len(nodes)):
        node_name = node_names[i]
        ini_content.append(f"#--{node_name}--")
        pppg_num = len(interface_flow_list[i])
        for j in range(pppg_num):
            ini_content.append(
                f"**.{node_name}.ppp[{j}].egressTC.mfClassifier.filters = xmldoc(\"filters_{node_name}{j}.xml\")")
            ini_content.append(f"**.{node_name}.ppp[{j}].egressTC.typename = \"PerFlowShaper\"")
            ini_content.append(f"**.{node_name}.ppp[{j}].egressTC.numFlows = {len(interface_flow_list[i][j])}")
            for k in range(len(interface_flow_list[i][j])):
                flow_idx = interface_flow_list[i][j][k]
                ini_content.append(
                    f"**.{node_name}.ppp[{j}].egressTC.tokenGenerator1[{k}].storageModule = \"^.bucket1[{k}].server\"")
                ini_content.append(
                    f"**.{node_name}.ppp[{j}].egressTC.tokenGenerator2[{k}].storageModule = \"^.bucket2[{k}].server\"")
                # Realize 2SRC using two token buckets:
                bk1_interval = round(1e6 / flow_profile[flow_idx][0] * option.packet_size, 3)
                bk1_burst = flow_profile[flow_idx][1] - flow_profile[flow_idx][0] * reprofiling_delay[flow_idx]
                bk1_token = round(bk1_burst / option.packet_size, 1) + 1
                if reprofiling_delay[flow_idx] == 0:
                    # The second token bucket uses the same parameters with the first if no reprofiling is needed.
                    bk2_interval = bk1_interval
                    bk2_token = bk1_token
                else:
                    bk2_rate = flow_profile[flow_idx][1] / reprofiling_delay[flow_idx]
                    bk2_interval = round(1e6 / bk2_rate * option.packet_size, 3)
                    bk2_token = 1
                bk_interval, bk_token = [bk1_interval, bk2_interval], [bk1_token, bk2_token]
                for bk, (interval, token) in enumerate(zip(bk_interval, bk_token)):
                    bk += 1
                    ini_content.append(
                        f"**.{node_name}.ppp[{j}].egressTC.tokenGenerator{bk}[{k}].generationInterval = {interval}us")
                    ini_content.append(
                        f"**.{node_name}.ppp[{j}].egressTC.bucket{bk}[{k}].server.maxNumTokens = {token}")
                    ini_content.append(
                        f"**.{node_name}.ppp[{j}].egressTC.bucket{bk}[{k}].server.initialNumTokens = {token}")
    ini_content.append("*.scenarioManager.script = xmldoc(\"scenario.xml\")")
    # Set the bandwidth at each link.
    scenario_content.append(f"    <at t=\"30\">")
    interface_idx = np.zeros((len(nodes),), dtype=int)
    for link, bandwidth in zip(links, link_bandwidth):
        src, dest = link[0], link[1]
        # Set the bandwidth if specified.
        if bandwidth[0] > 0:
            out_bw = f"{bandwidth[0]}bps"
            scenario_content.append(
                f"        <set-channel-param src-module=\"{node_names[src]}\" "
                f"src-gate=\"pppg$o[{interface_idx[src]}]\" par=\"datarate\" value=\"{out_bw}\"/>")
        if bandwidth[1] > 0:
            in_bw = f"{bandwidth[1]}bps"
            scenario_content.append(
                f"        <set-channel-param src-module=\"{node_names[src]}\" "
                f"src-gate=\"pppg$i[{interface_idx[src]}]\" par=\"datarate\" value=\"{in_bw}\"/>")
        interface_idx[src] += 1
        interface_idx[dest] += 1
    scenario_content.append(f"    </at>")
    # Write the composed ini file to the specified path.
    with open(os.path.join(option.out, option.ini_name + ".ini"), "w") as f:
        f.write("\n".join(ini_content))
    # Write the composed scenario manager file to the specified path.
    scenario_content.append("</scenario>")
    with open(os.path.join(option.out, "scenario.xml"), "w") as f:
        f.write("\n".join(scenario_content))
    return


def main(opts):
    # First, load the network data.
    network_data = np.load(opts.net)
    # Create the directory to save the simulation files.
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    # Load the flow path.
    flow_route = np.load(opts.route)
    # Load the flow profile.
    flow_data = np.load(opts.flow)
    flow_profile, per_hop = flow_data["flow"], flow_data["per_hop"]
    if per_hop:
        # Multiply end-to-end deadline by hop count when the loaded deadlines are specified as "per-hop".
        flow_profile[:, 2] = flow_profile[:, 2] * (np.sum(flow_route > 0, axis=1) - 1)
    # Load the reprofiling parameters.
    reprofiling_data = np.load(opts.reprofiling)
    # Load the traffic patterns.
    traffic_pattern = np.load(opts.traffic)
    # Collect the bandwidth of each link.
    bw_dict = {tuple(link): bw for link, bw in zip(reprofiling_data["link_map"], reprofiling_data["solution_"])}
    links = network_data["links"]
    link_bandwidth = np.zeros((len(links), 2))
    for link_idx, link in enumerate(links):
        link_bandwidth[link_idx, 0] = bw_dict.get(tuple(link), 0)
        link_bandwidth[link_idx, 1] = bw_dict.get(tuple(link[::-1]), 0)
    # Convert the flow profile datarate to Bps and link bandwidth to bps.
    if opts.datarate_unit == "bps":
        flow_profile[:, :2] /= 8
    elif opts.datarate_unit == "Bps":
        link_bandwidth *= 8
    elif opts.datarate_unit == "Kbps":
        flow_profile[:, :2] *= 125
        link_bandwidth *= 1000
    elif opts.datarate_unit == "KBps":
        flow_profile[:, :2] *= 1000
        link_bandwidth *= 8000
    elif opts.datarate_unit == "Mbps":
        flow_profile[:, :2] *= 1.25e5
        link_bandwidth *= 1e6
    elif opts.datarate_unit == "MBps":
        flow_profile[:, :2] *= 1e6
        link_bandwidth *= 8e6
    elif opts.datarate_unit == "Gbps":
        flow_profile[:, :2] *= 1.25e8
        link_bandwidth *= 1e9
    elif opts.datarate_unit == "GBps":
        flow_profile[:, :2] *= 1e9
        link_bandwidth *= 8e9
    else:
        raise Exception("Please set --datarate_unit from 'bps', 'Bps', 'Kbps', 'KBps', 'Mbps', 'MBps', 'Gbps', 'GBps'.")
    # Pre-process the flows.
    interface_flow_list = flow_count(network_data, flow_route, opts.out)
    # Compose the ini file.
    compose_ini(interface_flow_list, network_data, flow_route, flow_profile, reprofiling_data, link_bandwidth,
                traffic_pattern, opts)
    return


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('net', help="Path to the input npz file describing network nodes and links.")
    args.add_argument('route', help="Path to the input npy file describing route of the flows.")
    args.add_argument('flow', help="Path to the input npy file describing flow profiles.")
    args.add_argument('reprofiling', help="Path to the input npy file describing reprofiling parameters.")
    args.add_argument('traffic', help="Path to the input npy file describing traffic patterns.")
    args.add_argument('out', help="Directory to save results.")
    args.add_argument('--simulation_time', type=float, default=200, help="Total simulation time.")
    args.add_argument('--datarate_unit', type=str, default="KBps",
                      help="Unit of the reprofiling parameters. Available choices include 'bps', 'Bps', 'Kbps', 'KBps',"
                           "'Mbps', 'MBps', 'Gbps', 'GBps'.")
    args.add_argument('--packet_size', type=int, default=100,
                      help="Size of one packet in Bytes. Including 28 bytes of headers.")
    args.add_argument('--ned_name', type=str, default="test", help="Name of the ned file.")
    args.add_argument('--ini_name', type=str, default="test", help="Name of the ini file.")
    args.add_argument('--package_name', type=str, default="test", help="Package name of the ned file.")
    return args.parse_args()


if __name__ == "__main__":
    main(getargs())
