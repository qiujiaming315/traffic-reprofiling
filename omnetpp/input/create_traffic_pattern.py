import numpy as np
import os
from pathlib import Path

"""Generate traffic pattern input file for simulation."""

if __name__ == "__main__":
    # First, specify the directory and filename to save the generated inputs.
    path = "./traffic_pattern/"
    filename = "dc_traffic_pattern_10"
    # Next, specify an array with each row corresponding to the traffic pattern of a flow in the following format:
    # Each row (node) has four entries: start time of traffic.
    #                                   stop time of traffic.
    #                                   duration time for traffic on.
    #                                   duration time for traffic off.
    # traffic_pattern = np.array([[30, 200, 200, 10],
    #                             [30, 200, 200, 10]])
    traffic_pattern = np.ones((10, 4), dtype=int) * np.array([[30, 100, 100, 10]])
    # Create the directory and save the inputs.
    Path(path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(path, filename), traffic_pattern)
