import numpy as np
import os
import time

import sys
sys.path.insert(0, '/home/chiaraz/thesis/36_variables/liang_subsystems')
from compute_liang_subsystems import information_flow_subspace


def process_file(file_name, file_path):
    """
    Loads a file (assumed to be in N x 37 shape), deletes the first column (time/index),
    and returns the Liang subsystems analysis results.
    """
    full_path = os.path.join(file_path, file_name)
    data_in_file = []
    
    # Read file line by line
    with open(full_path, 'r') as f:
        for line in f:
            try:
                row = [float(x) for x in line.split()]
                if len(row) < 37:
                    # Skip lines with insufficient data
                    continue
                data_in_file.append(row)
            except ValueError:
                continue
    
    if not data_in_file:
        raise ValueError(f"No valid data in file {file_name}")
        
    data = np.array(data_in_file)
    
    # Determine if a transpose is needed.
    # We assume that the correct orientation is to have 36 rows (variables)
    # and many columns (observations). If the file is N x 37 with N >> 37,
    # then data.shape[0] > data.shape[1] and no transpose is needed.
    if data.shape[1] > data.shape[0]:
        time_series = data.T
    else:
        time_series = data

    # If the data still has 37 columns, assume the first column is time/index
    if time_series.shape[1] == 37:
        time_series = np.delete(time_series, 0, axis=1)
    
    # (Optional) Print shape for debugging:
    # print(f"Processed {file_name}: time_series shape = {time_series.shape}")
    
    # Set parameters (here r and s are chosen as in your original script)
    r = 20  # atmosphere variables (assumed)
    s = 36  # total number of variables after deleting the time column
    # Call the Liang analysis function.
    # Note: we pass r-1 and s-1 as in your original code.
    results = information_flow_subspace(time_series, r-1, s-1, np_val=1, n_iter=50, alpha=0.01)
    return results

def compute_stats(results_list, key):
    """
    Computes the mean and standard deviation for a given key from a list of result dictionaries.
    """
    values = [res[key] for res in results_list]
    return np.mean(values), np.std(values)

def main():
    start_time = time.time()
    
    # Set the folder where the 20 files are located.
    # Adjust this path as needed.
    file_path = '.'
    
    # Prepare lists of file names (without extensions) for fast and slow.
    fast_files = [f"{i}fast.txt" for i in range(1, 11)]
    slow_files = [f"{i}slow.txt" for i in range(1, 11)]
    
    fast_results = []
    slow_results = []
    
    print("Processing FAST files:")
    for fname in fast_files:
        print(f"Processing {fname} ...")
        try:
            res = process_file(fname, file_path)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue
        fast_results.append(res)
        print("  TAB:", res["TAB"])
        print("  TBA:", res["TBA"])
        print("  Error TAB:", res["error_TAB"])
        print("  Error TBA:", res["error_TBA"])
        print("  Significance TAB:", res["significance_TAB (bool, Z, p-value)"])
        print("  Significance TBA:", res["significance_TBA (bool, Z, p-value)"])
        print("-----")
        
    print("\nProcessing SLOW files:")
    for fname in slow_files:
        print(f"Processing {fname} ...")
        try:
            res = process_file(fname, file_path)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue
        slow_results.append(res)
        print("  TAB:", res["TAB"])
        print("  TBA:", res["TBA"])
        print("  Error TAB:", res["error_TAB"])
        print("  Error TBA:", res["error_TBA"])
        print("  Significance TAB:", res["significance_TAB (bool, Z, p-value)"])
        print("  Significance TBA:", res["significance_TBA (bool, Z, p-value)"])
        print("-----")
    
    elapsed = time.time() - start_time
    print("\nTotal processing time: {:.2f} seconds".format(elapsed))
    
    # Compute mean and standard deviation for key metrics in each category.
    # Here we compute for TAB and TBA as well as their errors.
    if fast_results:
        fast_TAB_mean, fast_TAB_std = compute_stats(fast_results, "TAB")
        fast_TBA_mean, fast_TBA_std = compute_stats(fast_results, "TBA")
        fast_error_TAB_mean, fast_error_TAB_std = compute_stats(fast_results, "error_TAB")
        fast_error_TBA_mean, fast_error_TBA_std = compute_stats(fast_results, "error_TBA")
        print("\n=== AVERAGE (FAST) ===")
        print("Fast atmosphere->ocean (TAB): mean = {:.4f}, std = {:.4f}".format(fast_TAB_mean, fast_TAB_std))
        print("Fast ocean->atmosphere (TBA): mean = {:.4f}, std = {:.4f}".format(fast_TBA_mean, fast_TBA_std))
        print("Fast Error TAB: mean = {:.4f}, std = {:.4f}".format(fast_error_TAB_mean, fast_error_TAB_std))
        print("Fast Error TBA: mean = {:.4f}, std = {:.4f}".format(fast_error_TBA_mean, fast_error_TBA_std))
    
    if slow_results:
        slow_TAB_mean, slow_TAB_std = compute_stats(slow_results, "TAB")
        slow_TBA_mean, slow_TBA_std = compute_stats(slow_results, "TBA")
        slow_error_TAB_mean, slow_error_TAB_std = compute_stats(slow_results, "error_TAB")
        slow_error_TBA_mean, slow_error_TBA_std = compute_stats(slow_results, "error_TBA")
        print("\n=== AVERAGE (SLOW) ===")
        print("Slow atmosphere->ocean (TAB): mean = {:.4f}, std = {:.4f}".format(slow_TAB_mean, slow_TAB_std))
        print("Slow ocean->atmosphere (TBA): mean = {:.4f}, std = {:.4f}".format(slow_TBA_mean, slow_TBA_std))
        print("Slow Error TAB: mean = {:.4f}, std = {:.4f}".format(slow_error_TAB_mean, slow_error_TAB_std))
        print("Slow Error TBA: mean = {:.4f}, std = {:.4f}".format(slow_error_TBA_mean, slow_error_TBA_std))

if __name__ == "__main__":
    main()
