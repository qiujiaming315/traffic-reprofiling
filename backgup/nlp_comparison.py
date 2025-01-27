import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import scipy.stats as st

from lib.network_parser import parse_link


def setup_axis():
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams['figure.figsize'] = (15, 10)
    ax = plt.subplot()
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.spines['top'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')
    ax.spines['left'].set_color('#606060')
    ax.spines['right'].set_color('#606060')
    ax.grid(True, color='#e9e9e9', linewidth=1)
    return ax


def plot_scatter(x_data, y_data, mask, labels, output_path, fig_name, percent_format=False, legend=True):
    ax = setup_axis()
    x_values, xlabel = x_data
    y_values, ylabel = y_data
    x_values, y_values = np.array(x_values), np.array(y_values)
    ax.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    ax.set_xlabel(xlabel, labelpad=15, color='#333333', size=40)
    if percent_format:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    x_values1, y_values1 = x_values[np.logical_not(mask)], y_values[np.logical_not(mask)]
    x_values2, y_values2 = x_values[mask], y_values[mask]
    ax.scatter(x_values1, y_values1, c="tab:blue", label=labels[0], s=60, alpha=0.3, edgecolors=None)
    ax.scatter(x_values2, y_values2, c="tab:red", label=labels[1], s=60, alpha=0.3, edgecolors=None)
    if legend:
        plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, fig_name + ".pdf"), bbox_inches='tight')
    plt.clf()
    return


def plot_yerr(x_data, y_data, labels, output_path, fig_name, y_err=None, percent_format=True, legend=True):
    ax = setup_axis()
    x_values, xlabel = x_data
    y_values, ylabel = y_data
    x_values, y_values = np.array(x_values), np.array(y_values)
    ax.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    ax.set_xlabel(xlabel, labelpad=15, color='#333333', size=40)
    if percent_format:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    num_line = len(y_values)
    colors = ["#7FB3D5", "#F7CAC9", "#A2C8B5", "#D9AFD9", "#D3D3D3", "#FFFF99", "#FFD1DC", "#9AD1D4", "#B19CD9",
              "#B0AFAF"]
    assert len(colors) >= num_line, "Too many lines to visualize."
    if y_err is None:
        for y_value, color, label in zip(y_values, colors, labels):
            ax.plot(x_values, y_value, 'o-', color=color, label=label, linewidth=5, markersize=15)
    else:
        for y_value, err, color, label in zip(y_values, y_err, colors, labels):
            ax.errorbar(x_values, y_value, err, fmt='o-', color=color, label=label, linewidth=5, markersize=15,
                        ecolor=color, elinewidth=5, capsize=9, capthick=5)
    if legend:
        plt.legend(fontsize=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, fig_name + ".pdf"), bbox_inches='tight')
    plt.clf()
    return


if __name__ == "__main__":
    data_dir = "../output/fifo/google/nlp_ipopt/"
    flow_dir = "../input/flow/google/"
    route_dir = "../input/route/google/"
    figure_dir = "../output/figures/fifo/google/nlp_comparison/"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    x_values = np.arange(10, 51, 10)
    x_data = [x_values, "Number of Flows"]
    nlp_fs_yvalue, nlp_fs_yerr = [], []
    nlp_fs_top_yvalue, nlp_fs_top_yerr = [], []
    nlp_ns_yvalue, nlp_ns_yerr = [], []
    nlp_ns_top_yvalue, nlp_ns_top_yerr = [], []
    fs_ns_yvalue, fs_ns_yerr = [], []
    nlp_time_yvalue, nlp_time_yerr = [], []
    fs_time_yvalue, fs_time_yerr = [], []
    nlp_fs_data, one_hop_ratio_data, avg_hop_num_flow_data, avg_hop_num_hop_data = [], [], [], []
    top_mask = np.array([], dtype=bool)
    for flow_num in x_values:
        data_file_dir = os.path.join(data_dir, str(flow_num))
        flow_file_dir = os.path.join(flow_dir, str(flow_num))
        route_file_dir = os.path.join(route_dir, str(flow_num))
        nlp_fs, fs_ns, nlp_ns, nlp_time, fs_time = [], [], [], [], []
        one_hop_ratio, avg_hop_num_flow, avg_hop_num_hop = [], [], []
        for file_idx in range(1000):
            file_name = os.path.join(data_file_dir, f"result{file_idx + 1}.npz")
            if os.path.isfile(file_name):
                file_data = np.load(file_name)
                nlp_fs.append((np.sum(file_data["fr"]) - np.sum(file_data["solution"])) / np.sum(file_data["fr"]) * 100)
                fs_ns.append((np.sum(file_data["nr"]) - np.sum(file_data["fr"])) / np.sum(file_data["nr"]) * 100)
                nlp_ns.append((np.sum(file_data["nr"]) - np.sum(file_data["solution"])) / np.sum(file_data["nr"]) * 100)
                nlp_time.append(file_data["time"])
                fs_time.append(file_data["fr_time"])
                # flow_name = os.path.join(flow_file_dir, f"flow{file_idx + 1}.npz")
                # flow_data = np.load(flow_name)['flow']
                route_name = os.path.join(route_file_dir, f"route{file_idx + 1}.npy")
                route_data = np.load(route_name)
                route_data, _, _ = parse_link(route_data)
                flow_num_hop = np.sum(route_data, axis=1)
                one_hop_ratio.append(np.sum(flow_num_hop == 1) / len(route_data))
                avg_hop_num_flow.append(np.sum(flow_num_hop) / len(route_data))
                num_hop_route = flow_num_hop[:, np.newaxis] * np.ones_like(route_data)
                num_hop_route = np.where(route_data, num_hop_route, 0)
                avg_hop_num_hop.append(
                    np.sum(np.sum(num_hop_route, axis=0) / np.sum(route_data, axis=0)) / route_data.shape[1])
        nlp_fs_thresh = np.sort(nlp_fs)[int(len(nlp_fs) * 0.95)]
        nlp_fs_top = [a for a in nlp_fs if a >= nlp_fs_thresh]
        nlp_fs_yvalue.append(np.mean(nlp_fs))
        nlp_fs_top_yvalue.append(np.mean(nlp_fs_top))
        nlp_ns_yvalue.append(np.mean(nlp_ns))
        fs_ns_yvalue.append(np.mean(fs_ns))
        nlp_time_yvalue.append(np.mean(nlp_time))
        fs_time_yvalue.append(np.mean(fs_time))
        fs_err = st.norm.interval(0.95, loc=np.mean(nlp_fs), scale=st.sem(nlp_fs))
        fs_top_err = st.t.interval(0.95, df=len(nlp_fs_top) - 1, loc=np.mean(nlp_fs_top), scale=st.sem(nlp_fs_top))
        nlp_err = st.norm.interval(0.95, loc=np.mean(nlp_ns), scale=st.sem(nlp_ns))
        ns_err = st.norm.interval(0.95, loc=np.mean(fs_ns), scale=st.sem(fs_ns))
        time_err = st.norm.interval(0.95, loc=np.mean(nlp_time), scale=st.sem(nlp_time))
        fs_time_err = st.norm.interval(0.95, loc=np.mean(fs_time), scale=st.sem(fs_time))
        nlp_fs_yerr.append((fs_err[1] - fs_err[0]) / 2)
        nlp_fs_top_yerr.append((fs_top_err[1] - fs_top_err[0]) / 2)
        nlp_ns_yerr.append((nlp_err[1] - nlp_err[0]) / 2)
        fs_ns_yerr.append((ns_err[1] - ns_err[0]) / 2)
        nlp_time_yerr.append((time_err[1] - time_err[0]) / 2)
        fs_time_yerr.append((fs_time_err[1] - fs_time_err[0]) / 2)
        nlp_fs_data.extend(nlp_fs)
        one_hop_ratio_data.extend(one_hop_ratio)
        avg_hop_num_flow_data.extend(avg_hop_num_flow)
        avg_hop_num_hop_data.extend(avg_hop_num_hop)
        top_mask = np.append(top_mask, nlp_fs >= nlp_fs_thresh)
    nlp_fs_ydata = [[nlp_fs_yvalue, nlp_fs_top_yvalue], "Relative Improvement"]
    nlp_ns_ydata = [[nlp_ns_yvalue], "Relative Improvement"]
    fs_ns_ydata = [[fs_ns_yvalue], "Relative Improvement"]
    time_ydata = [[nlp_time_yvalue, fs_time_yvalue], "Time Taken (seconds)"]
    plot_yerr(x_data, nlp_fs_ydata, ["overall", "top 5%"], figure_dir, "nlp_fr", y_err=[nlp_fs_yerr, nlp_fs_top_yerr],
              percent_format=True, legend=True)
    plot_yerr(x_data, nlp_ns_ydata, [""], figure_dir, "nlp_nr", y_err=[nlp_ns_yerr], percent_format=True, legend=False)
    plot_yerr(x_data, fs_ns_ydata, [""], figure_dir, "fr_nr", y_err=[fs_ns_yerr], percent_format=True, legend=False)
    plot_yerr(x_data, time_ydata, ["NLP", "FR"], figure_dir, "time", y_err=[nlp_time_yerr, fs_time_yerr],
              percent_format=False, legend=True)
    plot_yerr(x_data, [[nlp_time_yvalue], "Time Taken (seconds)"], [""], figure_dir, "time_new", y_err=[nlp_time_yerr],
              percent_format=False, legend=False)
    # scatter_ydata = [nlp_fs_data, "Relative Improvement to FS"]
    # one_hop_ratio_xdata = [one_hop_ratio_data, "Percentage of One-Hop Flows"]
    # avg_hop_num_flow_xdata = [avg_hop_num_flow_data, "Average Number of Hops Traversed"]
    # avg_hop_num_hop_xdata = [avg_hop_num_hop_data, "Flow Span Index"]
    # plot_scatter(one_hop_ratio_xdata, scatter_ydata, top_mask, ["overall", "top 5%"], figure_dir, "one_hop_ratio",
    #              percent_format=True)
    # plot_scatter(avg_hop_num_flow_xdata, scatter_ydata, top_mask, ["overall", "top 5%"], figure_dir, "avg_num_hop",
    #              percent_format=True)
    # plot_scatter(avg_hop_num_hop_xdata, scatter_ydata, top_mask, ["overall", "top 5%"], figure_dir, "flow_span_index",
    #              percent_format=True)
