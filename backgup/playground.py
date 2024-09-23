import numpy as np
import os
import re

# string = "C12_ct0"
# if re.match(r"C[0-9]+_ct[0-9]+", string):
#     print("match")

file_dir = "../output/sced/tandem/nlp_octeract/3_5/"
file_dir1 = "../output/sced/tandem/nlp_ipopt/3_5/"
file_name = "result140-150.npz"

file_path = os.path.join(file_dir, file_name)
file_path1 = os.path.join(file_dir1, file_name)
file_data = np.load(file_path)
file_data1 = np.load(file_path1)
check = np.all(np.abs(file_data["solution"] - file_data1["solution"]))
print()

# flow_dir = "../input/flow/tandem/line1/3/"
# flow_files = sorted(os.listdir(flow_dir))
# print()
