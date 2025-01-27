import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import scipy.stats as st


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
    data_dir = "../output/fifo/google/baseline/"
    data_dir_sced = "../output/sced/google/greedy/"
    flow_dir = "../input/flow/google/"
    route_dir = "../input/route/google/"
    figure_dir = "../output/figures/fifo/google/nlp_comparison/"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    x_values = np.arange(100, 1001, 100)
    x_data = [x_values, "Number of Flows"]
    fifo_sced_yvalue, fifo_sced_yerr = [], []
    for flow_num in x_values:
        data_file_dir = os.path.join(data_dir, str(flow_num))
        data_file_dir_sced = os.path.join(data_dir_sced, str(flow_num))
        fifo_sced = []
        for file_idx in range(1000):
            file_name = os.path.join(data_file_dir, f"result{file_idx + 1}.npz")
            file_name_sced = os.path.join(data_file_dir_sced, f"result{file_idx + 1}.npz")
            if os.path.isfile(file_name):
                file_data = np.load(file_name)
                file_data_sced = np.load(file_name_sced)
                fifo_sced.append(
                    (np.sum(file_data["fr"]) - np.sum(file_data_sced["solution"])) / np.sum(file_data["fr"]) * 100)
                # flow_name = os.path.join(flow_file_dir, f"flow{file_idx + 1}.npz")
                # flow_data = np.load(flow_name)['flow']
        fifo_sced_yvalue.append(np.mean(fifo_sced))
        sced_err = st.norm.interval(0.95, loc=np.mean(fifo_sced), scale=st.sem(fifo_sced))
        fifo_sced_yerr.append((sced_err[1] - sced_err[0]) / 2)
    fifo_sced_ydata = [[fifo_sced_yvalue], "Relative Improvement"]
    plot_yerr(x_data, fifo_sced_ydata, [""], figure_dir, "fifo_fr_sced_greedy", y_err=[fifo_sced_yerr],
              percent_format=True, legend=False)
