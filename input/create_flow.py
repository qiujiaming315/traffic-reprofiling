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
fb_size = fb_size / 1000
fb_duration = np.concatenate((fb_web_duration, fb_cache_duration, fb_hadoop_duration), axis=0)
fb_duration = fb_duration / 1000
fb_deadline = np.array([0.01, 0.05, 0.2])
fb_ratio = np.array([3, 9, 1])
fb_burst = np.array([0.15, 0.4, 0.3]) / 1000
fb_burst_scale = np.array([5, 10, 1])
fb_burst = fb_burst * fb_burst_scale
fb_ratio = fb_ratio / np.sum(fb_ratio)

chameleon_ia_rate1 = np.array([[300, 550]])
chameleon_ia_rate2 = np.array([[150, 550]])
chameleon_ia_rate3 = np.array([[100, 500]])
chameleon_ia_rate4 = np.array([[1, 100]])
chameleon_ia_rate = np.concatenate((chameleon_ia_rate1, chameleon_ia_rate2, chameleon_ia_rate3, chameleon_ia_rate4),
                                   axis=0) / 1000
chameleon_ia_burst1 = np.array([[100, 400]])
chameleon_ia_burst2 = np.array([[100, 400]])
chameleon_ia_burst3 = np.array([[100, 400]])
chameleon_ia_burst4 = np.array([[80, 120]])
chameleon_ia_burst = np.concatenate(
    (chameleon_ia_burst1, chameleon_ia_burst2, chameleon_ia_burst3, chameleon_ia_burst4),
    axis=0) / 1e6
chameleon_ia_deadline1 = np.array([[80, 120]])
chameleon_ia_deadline2 = np.array([[150, 200]])
chameleon_ia_deadline3 = np.array([[10, 20]])
chameleon_ia_deadline4 = np.array([[10, 20]])
chameleon_ia_deadline = np.concatenate(
    (chameleon_ia_deadline1, chameleon_ia_deadline2, chameleon_ia_deadline3, chameleon_ia_deadline4),
    axis=0) / 1000
chameleon_cs_rate = np.array([[1, 220]]) / 1000
chameleon_cs_burst = np.array([[80, 300]]) / 1e6
chameleon_cs_deadline = np.array([[2, 4]]) / 1000
chameleon_cps_rate1 = np.array([[2, 4]])
chameleon_cps_rate2 = np.array([[5, 8]])
chameleon_cps_rate3 = np.array([[2, 4]])
chameleon_cps_rate = np.concatenate((chameleon_cps_rate1, chameleon_cps_rate2, chameleon_cps_rate3), axis=0)
chameleon_cps_burst1 = np.array([[80, 140]])
chameleon_cps_burst2 = np.array([[1000, 3000]])
chameleon_cps_burst3 = np.array([[80, 120]])
chameleon_cps_burst = np.concatenate((chameleon_cps_burst1, chameleon_cps_burst2, chameleon_cps_burst3), axis=0) / 1e6
chameleon_cps_deadline1 = np.array([[50, 200]])
chameleon_cps_deadline2 = np.array([[50, 200]])
chameleon_cps_deadline3 = np.array([[50, 200]])
chameleon_cps_deadline = np.concatenate((chameleon_cps_deadline1, chameleon_cps_deadline2, chameleon_cps_deadline3),
                                        axis=0) / 1000
chameleon_bh_rate1 = np.array([[100, 150]])
chameleon_bh_rate2 = np.array([[100, 200]])
chameleon_bh_rate3 = np.array([[80, 200]])
chameleon_bh_rate = np.concatenate((chameleon_bh_rate1, chameleon_bh_rate2, chameleon_bh_rate3), axis=0)
chameleon_bh_burst1 = np.array([[1000, 5000]])
chameleon_bh_burst2 = np.array([[1000, 3000]])
chameleon_bh_burst3 = np.array([[1000, 3000]])
chameleon_bh_burst = np.concatenate((chameleon_bh_burst1, chameleon_bh_burst2, chameleon_bh_burst3), axis=0) / 1e6
chameleon_bh_deadline1 = np.array([[10, 100]])
chameleon_bh_deadline2 = np.array([[10, 100]])
chameleon_bh_deadline3 = np.array([[50, 100]])
chameleon_bh_deadline = np.concatenate((chameleon_bh_deadline1, chameleon_bh_deadline2, chameleon_bh_deadline3),
                                       axis=0) / 1000
chameleon_ratio = np.array([2, 5, 2, 1])
chameleon_ratio = chameleon_ratio / np.sum(chameleon_ratio)

tsn_cdt_interval = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tsn_a_interval = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tsn_b_interval = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tsn_interval = [tsn_cdt_interval, tsn_a_interval, tsn_b_interval]
tsn_interval = [class_interval / 1000 for class_interval in tsn_interval]
tsn_frame_size = np.array([128, 256, 256]) / 1e6
tsn_deadline = np.array([0.1, 2, 50]) / 1000
tsn_ratio = np.array([1, 4, 4])
tsn_ratio = tsn_ratio / np.sum(tsn_ratio)


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
    The paper is available at https://dl.acm.org/doi/10.1145/2785956.2787472.
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


def generate_chameleon_flow(num_flow, seed=None):
    """
    Generate random flow profiles using distribution reported in the Chameleon paper.
    The paper is available at https://dl.acm.org/doi/10.1145/3386367.3432879.
    :param num_flow: the number of flows in the generated profile.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the flow profile.
    """
    flow = np.zeros((num_flow, 3))
    rstate = np.random.RandomState(seed)
    application_rate = [chameleon_ia_rate, chameleon_cs_rate, chameleon_cps_rate, chameleon_bh_rate]
    application_burst = [chameleon_ia_burst, chameleon_cs_burst, chameleon_cps_burst, chameleon_bh_burst]
    application_deadline = [chameleon_ia_deadline, chameleon_cs_deadline, chameleon_cps_deadline, chameleon_bh_deadline]
    # Sanity check on the parameters:
    for r, b, d in zip(application_rate, application_burst, application_deadline):
        assert len(r) == len(b) and len(b) == len(d)
    # Sample application category according to the sampling ratio.
    application_mask = rstate.choice(len(chameleon_ratio), num_flow, p=chameleon_ratio)
    # Sample flow profiles for each application category.
    for app_category_idx in range(len(chameleon_ratio)):
        num_app = np.sum(application_mask == app_category_idx)
        app_data = np.zeros((num_app, 3))
        # Sample applications from the specified category.
        app_mask = rstate.choice(len(application_rate[app_category_idx]), num_app)
        for app_idx in range(len(application_rate[app_category_idx])):
            # Set the rate, burst and deadline.
            num_sample = np.sum(app_mask == app_idx)
            app_rate = application_rate[app_category_idx][app_idx]
            app_burst = application_burst[app_category_idx][app_idx]
            app_deadline = application_deadline[app_category_idx][app_idx]
            rand_data = rstate.rand(num_sample, 3)
            app_data[app_mask == app_idx, 0] = rand_data[:, 0] * (app_rate[1] - app_rate[0]) + app_rate[0]
            app_data[app_mask == app_idx, 1] = rand_data[:, 1] * (app_burst[1] - app_burst[0]) + app_burst[0]
            app_data[app_mask == app_idx, 2] = rand_data[:, 2] * (app_deadline[1] - app_deadline[0]) + app_deadline[0]
        flow[application_mask == app_category_idx] = app_data
    return flow


def generate_tsn_flow(num_flow, periodic=True, seed=None):
    """
    Generate random flow profiles for TSN applications.
    Motivated by the following papers:
    https://ieeexplore.ieee.org/abstract/document/7092358/.
    https://ieeexplore.ieee.org/abstract/document/8700610/.
    https://ieeexplore.ieee.org/abstract/document/7385584/.
    :param num_flow: the number of flows in the generated profile.
    :param periodic: whether the arrival process is periodic with jitter or Poisson.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the flow profile.
    """
    flow = np.zeros((num_flow, 3))
    rstate = np.random.RandomState(seed)
    # Sample traffic class according to the sampling ratio.
    class_mask = rstate.choice(len(tsn_ratio), num_flow, p=tsn_ratio)
    # Sample flow profiles for each traffic class.
    for class_idx in range(len(tsn_ratio)):
        num_flow = np.sum(class_mask == class_idx)
        flow_data = np.zeros((num_flow, 3))
        # Sample arrival interval from the specified traffic class.
        interval_mask = rstate.choice(len(tsn_interval[class_idx]), num_flow)
        # Set the rate, burst and deadline.
        flow_data[:, 0] = tsn_frame_size[class_idx] / tsn_interval[class_idx][interval_mask]
        flow_data[:, 1] = tsn_frame_size[class_idx]
        if not periodic:
            flow_data[:, 0] *= 1.1
            flow_data[:, 1] *= 25
        flow_data[:, 2] = tsn_deadline[class_idx]
        flow[class_mask == class_idx] = flow_data
    return flow


def save_file(output_path, file_name, flow, per_hop=False):
    """
    Save the generated flow profile to the specified output location.
    :param output_path: the directory to save the flow profile.
    :param file_name: name of the file to save the flow profile.
    :param flow: the flow profile.
    :param per_hop: whether the deadline stands for per hop deadline (True) or end-to-end deadline (False).
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_path, file_name + ".npz"), flow=flow, per_hop=per_hop)
    return


if __name__ == "__main__":
    # First, specify the directory to save the generated flow profiles.
    path = f"./flow/"
    # You can specify your own flow profile and directly save it to the directory.
    flow = np.array([[15.0, 8.0, 1.0],
                     [2.0, 4.0, 2.0],
                     [3.0, 13.0, 5.0]
                     ])
    save_file(path, flow)
    # Alternatively, you may generate and save a random flow profile.
    save_file(path, generate_random_flow(10))
    # Or you can generate a flow profile motivated by the Facebook paper (for inter-datacenter network).
    save_file(path, generate_fb_flow(10))
    # Or you can generate a flow profile motivated by the Chameleon paper (for intra-datacenter network).
    save_file(path, generate_chameleon_flow(10))
    # Or you can generate a flow profile for TSN network.
    save_file(path, generate_tsn_flow(116))
