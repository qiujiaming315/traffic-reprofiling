import numpy as np
import os
from pathlib import Path

"""Generate flow profiles (rate, burst, end-to-end deadline) as optimization input."""
cdf_values = np.arange(0, 1.01, 0.05)
fb_web_size = np.array(
    [[0.07, 0.15, 0.15, 0.2, 0.3, 0.3, 0.3, 0.5, 0.6, 0.75, 0.85, 1.2, 2, 2.9, 3.8, 7, 15, 23, 40, 90, 200]])
# fb_cache_size = np.array(
#     [[0.07, 0.15, 0.25, 0.35, 0.4, 0.5, 0.6, 0.7, 0.9, 1.5, 2, 2, 2.4, 3, 5, 7, 10, 10.4, 11, 1000, 1100]])
fb_cache_size = np.array(
    [[0.07, 0.1, 0.4, 0.8, 2, 2.2, 2.5, 2.7, 3, 3.5, 3.8, 4, 4.5, 4.8, 5, 5.8, 7, 80, 1000, 1500, 3000]])
fb_hadoop_size = np.array(
    [[0.08, 0.15, 0.22, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.7, 1.5, 2.7, 10, 300]])
fb_web_duration = np.array(
    [[1, 1, 1, 2, 4, 40, 90, 200, 600, 2000, 10500, 20000, 40000, 60000, 80000, 100000, 130000, 140000, 150000, 150000,
      150000]])
fb_cache_duration = np.array(
    [[1, 1, 1, 100, 2700, 40000, 70000, 80000, 85000, 90000, 100000, 105000, 110000, 120000, 130000, 200000, 500000,
      550000, 600000, 600000, 600000]])
fb_hadoop_duration = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 300, 3500, 15000, 35000, 100000, 600000]])
fb_size = np.concatenate((fb_web_size, fb_cache_size, fb_hadoop_size), axis=0)
fb_duration = np.concatenate((fb_web_duration, fb_cache_duration, fb_hadoop_duration), axis=0)
fb_duration = fb_duration / 1000
fb_deadline = np.array([0.01, 0.05, 0.2])
fb_ratio = np.array([3, 9, 1])
fb_burst = np.array([0.15, 0.4, 0.3])
fb_burst_scale = np.array([5, 10, 1])
fb_burst = fb_burst * fb_burst_scale
fb_ratio = fb_ratio / np.sum(fb_ratio)


def generate_random_flow(num_flow, seed=None):
    """
    Generate random flow profiles.
    :param num_flow: the number of flows in the generated profile.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the flow profile.
    """
    # Set the bounds for randomly generating flow profiles.
    rate_bound = (1, 100)
    burst_bound = (1, 100)
    # Configurations of deadline classes (uncomment the second line to switch from configuration 1 to 2).
    deadline_class = np.array([0.01, 0.1, 1])
    # deadline_class = np.array([0.01, 0.025, 0.05, 0.1])
    rstate = np.random.RandomState(seed)
    flow = np.zeros((num_flow, 3))
    rand_data = rstate.rand(num_flow, 2)
    # Randomly select the rate, burst, and deadline for each flow.
    flow[:, 0] = np.around(rand_data[:, 0] * (rate_bound[1] - rate_bound[0]) + rate_bound[0], 2)
    flow[:, 1] = np.around(rand_data[:, 1] * (burst_bound[1] - burst_bound[0]) + burst_bound[0], 2)
    flow[:, 2] = deadline_class[rstate.randint(len(deadline_class), size=num_flow)]
    return flow


def generate_fb_flow(num_flow, seed=None):
    """
    Generate random flow profiles using distribution reported in the Facebook paper.
    :param num_flow: the number of flows in the generated profile.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the flow profile.
    """
    flow = np.zeros((num_flow, 3))
    int_num = len(cdf_values) - 1
    rstate = np.random.RandomState(seed)
    # Select the application type for each flow.
    flow_type = rstate.choice(3, num_flow, p=fb_ratio)
    # Randomly select the rate, burst, and deadline for each flow according to the Facebook flow distributions.
    int_point = rstate.randint(int_num, size=num_flow)
    size1, size2 = fb_size[flow_type, int_point], fb_size[flow_type, int_point + 1]
    duration1, duration2 = fb_duration[flow_type, int_point], fb_duration[flow_type, int_point + 1]
    rand_data = rstate.rand(2, num_flow)
    size = (size2 - size1) * rand_data[0] + size1
    duration = (duration2 - duration1) * rand_data[0] + duration1
    flow[:, 0] = size / duration
    flow[:, 1] = fb_burst[flow_type] * 2 * rand_data[1]
    flow[:, 2] = fb_deadline[flow_type]
    return flow


def save_file(output_path, flow, per_hop=False):
    """
    Save the generated flow profile to the specified output location.
    :param output_path: the directory to save the flow profile.
    :param flow: the flow profile.
    :param per_hop: whether the deadline stands for per hop deadline (True) or end-to-end deadline (False).
    """
    num_flow = flow.shape[0]
    output_path = os.path.join(output_path, str(num_flow), "")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.savez(f"{output_path}flow{num_files + 1}.npz", flow=flow, per_hop=per_hop)
    return


if __name__ == "__main__":
    path = f"./flow/"
    save_file(path, generate_fb_flow(int(10)))
