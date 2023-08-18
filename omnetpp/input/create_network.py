import numpy as np
import os
from pathlib import Path

"""Generate network topology as an input file for simulation."""
google_nodes = np.array([[1, 150, 160],
                         [1, 150, 220],
                         [1, 170, 280],
                         [1, 270, 250],
                         [1, 300, 280],
                         [1, 420, 220],
                         [1, 440, 190],
                         [1, 420, 280],
                         [1, 460, 220],
                         [1, 460, 160],
                         [1, 420, 340],
                         [0, 130, 160],
                         [0, 130, 220],
                         [0, 150, 280],
                         [0, 270, 230],
                         [0, 300, 300],
                         [0, 420, 200],
                         [0, 440, 170],
                         [0, 440, 280],
                         [0, 480, 220],
                         [0, 480, 160],
                         [0, 440, 340]
                         ])
google_links = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (3, 7), (4, 5), (4, 7),
                (4, 10), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (6, 9), (7, 8), (7, 10), (8, 9), (0, 11), (1, 12),
                (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21)]
cev_nodes = np.array([[1, 150, 190],
                      [1, 150, 280],
                      [1, 160, 160],
                      [1, 160, 310],
                      [1, 180, 220],
                      [1, 180, 260],
                      [1, 290, 220],
                      [1, 290, 260],
                      [1, 390, 180],
                      [1, 390, 290],
                      [1, 470, 180],
                      [1, 450, 240],
                      [1, 470, 290],
                      [0, 100, 180],
                      [0, 100, 200],
                      [0, 100, 240],
                      [0, 100, 260],
                      [0, 100, 280],
                      [0, 100, 300],
                      [0, 100, 320],
                      [0, 120, 120],
                      [0, 160, 120],
                      [0, 200, 120],
                      [0, 140, 340],
                      [0, 180, 340],
                      [0, 210, 180],
                      [0, 210, 290],
                      [0, 210, 320],
                      [0, 250, 180],
                      [0, 290, 180],
                      [0, 330, 180],
                      [0, 250, 290],
                      [0, 290, 290],
                      [0, 330, 290],
                      [0, 370, 150],
                      [0, 410, 150],
                      [0, 370, 330],
                      [0, 410, 330],
                      [0, 450, 150],
                      [0, 490, 150],
                      [0, 490, 230],
                      [0, 490, 250],
                      [0, 450, 330],
                      [0, 490, 330]
                      ])
cev_links = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (6, 11), (7, 9),
             (7, 11), (8, 10), (9, 12), (0, 13), (0, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (2, 20), (2, 21),
             (2, 22), (3, 23), (3, 24), (4, 25), (5, 26), (5, 27), (6, 28), (6, 29), (6, 30), (7, 31), (7, 32), (7, 33),
             (8, 34), (8, 35), (9, 36), (9, 37), (10, 38), (10, 39), (11, 40), (11, 41), (12, 42), (12, 43)]
cev_nodes_condensed = np.array([[1, 150, 190],
                                [1, 150, 280],
                                [1, 160, 160],
                                [1, 160, 310],
                                [1, 180, 220],
                                [1, 180, 260],
                                [1, 290, 220],
                                [1, 290, 260],
                                [1, 390, 180],
                                [1, 390, 290],
                                [1, 470, 180],
                                [1, 450, 240],
                                [1, 470, 290],
                                [0, 100, 190],
                                [0, 100, 280],
                                [0, 160, 120],
                                [0, 160, 340],
                                [0, 210, 180],
                                [0, 210, 290],
                                [0, 290, 180],
                                [0, 290, 290],
                                [0, 390, 150],
                                [0, 390, 330],
                                [0, 470, 150],
                                [0, 490, 240],
                                [0, 470, 330]
                                ])
cev_links_condensed = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (6, 11),
                       (7, 9), (7, 11), (8, 10), (9, 12), (0, 13), (1, 14), (2, 15), (3, 16), (4, 17), (5, 18), (6, 19),
                       (7, 20), (8, 21), (9, 22), (10, 23), (11, 24), (12, 25)]


def generate_node_name(nodes):
    """
    Generate automatic node names.
    :param nodes: the node array.
    :return: a list of node names.
    """
    node_names = list()
    host_count, router_count = 0, 0
    for node_type in nodes[:, 0]:
        if node_type == 0:
            host_count += 1
            name_prefix, name_suffix = 'H', ""
            d = host_count
            while d > 0:
                d, m = divmod(d - 1, 26)
                name_suffix = chr(ord('A') + m) + name_suffix
        else:
            router_count += 1
            name_prefix, name_suffix = 'R', ""
            d = router_count
            while d > 0:
                d, m = divmod(d - 1, 26)
                name_suffix = chr(ord('A') + m) + name_suffix
        node_names.append(name_prefix + name_suffix)
    return node_names


if __name__ == "__main__":
    # First, specify the directory and filename to save the generated inputs.
    path = "./network/"
    filename = "dc_net"
    # Next, specify an array with each row corresponding to a network node in the following format:
    # Each row (node) has three entries: a binary (with 0 indicating an end-host and 1 indicating a router)
    #                                    the x-axis of the node to visualize in the simulator.
    #                                    the y-axis of the node to visualize in the simulator.
    # nodes = np.array([[1, 365, 251],
    #                   [1, 440, 321],
    #                   [1, 514, 251],
    #                   [1, 440, 181],
    #                   [0, 263, 159],
    #                   [0, 337, 421],
    #                   [0, 612, 339],
    #                   [0, 536, 88]
    #                   ])
    # nodes = np.array([[1, 301, 241],
    #                   [1, 449, 241],
    #                   [0, 173, 241],
    #                   [0, 585, 241]
    #                   ])
    nodes = google_nodes
    # Next, specify a list of names, one for each node.
    # You can generate automatic node names.
    node_names = generate_node_name(nodes)
    # Next, specify a list of (bidirectional) links.
    # links = [(0, 2), (1, 3), (0, 1)]
    links = google_links
    # Create the directory and save the inputs.
    Path(path).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(path, filename), nodes=nodes, node_names=node_names, links=links)
