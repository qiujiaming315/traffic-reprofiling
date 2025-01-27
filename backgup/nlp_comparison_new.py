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
    ax.tick_params(axis='x', labelsize=40)
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
    data_dir = "../output/fifo/google/nlp_ipopt/50/"
    figure_dir = "../output/figures/fifo/google/nlp_comparison/"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    top_mask = np.array([], dtype=bool)
    gs_bw, nlp_bw, ntb_bw = [], [], []
    for file_idx in range(1000):
        file_name = os.path.join(data_dir, f"result{file_idx + 1}.npz")
        if os.path.isfile(file_name):
            file_data = np.load(file_name)
            gs_bw.append(np.sum(file_data["fr"]))
            nlp_bw.append(np.sum(file_data["solution"]))
            ntb_bw.append(np.sum(file_data["nr"]))
    nlp_bw_avg = np.mean(nlp_bw)
    gs_bw = np.array(gs_bw) / nlp_bw_avg
    nlp_bw = np.array(nlp_bw) / nlp_bw_avg
    ntb_bw = np.array(ntb_bw) / nlp_bw_avg
    gs_bw_avg, nlp_bw_avg, ntb_bw_avg = np.mean(gs_bw), np.mean(nlp_bw), np.mean(ntb_bw)
    gs_err = st.norm.interval(0.95, loc=np.mean(gs_bw), scale=st.sem(gs_bw))
    nlp_err = st.norm.interval(0.95, loc=np.mean(nlp_bw), scale=st.sem(nlp_bw))
    ntb_err = st.norm.interval(0.95, loc=np.mean(ntb_bw), scale=st.sem(ntb_bw))
    print()
    # labels = ["GS", "NLP", "NTB"]
    # plot_data = [[gs_bw, nlp_bw, ntb_bw], "Normalized Bandwidth"]
    # plot_box(plot_data, labels, figure_dir, "shaping_bw_box")
