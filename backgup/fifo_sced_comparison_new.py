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


def plot_box(y_data, labels, output_path, fig_name):
    ax = setup_axis()
    ydata, ylabel = y_data
    ax.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    ax.boxplot(ydata, patch_artist=True, tick_labels=labels)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, fig_name + ".pdf"), bbox_inches='tight')
    plt.clf()
    return


if __name__ == "__main__":
    data_dir = "../output/fifo/google/baseline/10000/"
    data_dir_sced = "../output/sced/google/greedy/10000/"
    figure_dir = "../output/figures/fifo/google/nlp_comparison/"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    fifo_bw, sced_bw = [], []
    for file_idx in range(1000):
        file_name = os.path.join(data_dir, f"result{file_idx + 1}.npz")
        file_name_sced = os.path.join(data_dir_sced, f"result{file_idx + 1}.npz")
        if os.path.isfile(file_name):
            file_data = np.load(file_name)
            file_data_sced = np.load(file_name_sced)
            fifo_bw.append(np.sum(file_data["fr"]))
            sced_bw.append(np.sum(file_data_sced["solution"]))
    fifo_sced = [(f - s) / f for f, s in zip(fifo_bw, sced_bw)]
    fifo_sced_avg = np.mean(fifo_sced)
    fifo_bw_avg = np.mean(fifo_bw)
    fifo_bw = np.array(fifo_bw) / fifo_bw_avg
    sced_bw = np.array(sced_bw) / fifo_bw_avg
    fifo_bw_avg, sced_bw_avg = np.mean(fifo_bw), np.mean(sced_bw)
    fifo_err = st.norm.interval(0.95, loc=np.mean(fifo_bw), scale=st.sem(fifo_bw))
    sced_err = st.norm.interval(0.95, loc=np.mean(sced_bw), scale=st.sem(sced_bw))
    print()
    # labels = ["FIFO", "SCED"]
    # plot_data = [[fifo_bw, sced_bw], "Normalized Bandwidth"]
    # plot_box(plot_data, labels, figure_dir, "scheduling_bw_box")
