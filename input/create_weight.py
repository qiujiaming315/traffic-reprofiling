import numpy as np
import os
from pathlib import Path

"""Generate link bandwidth weights as input file for optimization."""


def save_file(output_path, weight):
    """Save the generated network routes to the specified output location."""
    num_link = len(weight)
    output_path = os.path.join(output_path, str(num_link), "")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    num_files = len(os.listdir(output_path))
    np.save(f"{output_path}weight{num_files + 1}.npy", weight)
    return


if __name__ == "__main__":
    output_path = "./weight/"
    weight = np.array([2.0, 1.0, 1.0])
    save_file(output_path, weight)
