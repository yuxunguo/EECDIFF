import re
import numpy as np
import csv

input_file = "ee_EEC_data/Raw Data/Aleph/points_Wenbin.txt"   # your input data file
output_file = "ee_EEC_data/Raw Data/Aleph/alephEEC.csv"

# regex patterns to match numbers, sys, and stat errors
num_pattern = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
sys_pattern = re.compile(r"sys:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)")
stat_pattern = re.compile(r"stat:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)")

rows = []

with open(input_file, "r") as f:
    for line in f:
        # find all numeric values (first two = z and EEC)
        nums = num_pattern.findall(line)
        if len(nums) < 2:
            continue

        z = float(nums[0])
        eec = float(nums[1])

        # extract sys and stat if they exist
        sys_match = sys_pattern.search(line)
        stat_match = stat_pattern.search(line)

        sys_err = float(sys_match.group(1)) if sys_match else 0.0
        stat_err = float(stat_match.group(1)) if stat_match else 0.0

        rows.append([z, eec, sys_err, stat_err])

# write to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["z", "EEC", "sys_err", "stat_err"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to '{output_file}'.")