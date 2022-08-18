import numpy as np
import os
from pathlib import Path

"""Generate the weight of each link bandwidth as optimization input."""


def save_file(output_path, weight):
    """
    Save the generated weights to the specified output location.
    :param output_path: the directory to save the weights.
    :param weight: the weights.
    """
    num_link = len(weight)
    output_path = os.path.join(output_path, str(num_link), "")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.save(f"{output_path}weight{num_files + 1}.npy", weight)
    return


if __name__ == "__main__":
    # First, specify the directory to save the generated link weights.
    path = "./weight/"
    # You should then specify the weight you want to assign on each link and save it to the specified directory.
    wt = np.array([2.0, 1.0, 1.0])
    save_file(path, wt)
