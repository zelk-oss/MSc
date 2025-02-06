import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import stats

# Load multiple files
#file_paths = glob.glob("/home/chiaraz/thesis/36_variables/averaging/results_averaging_atmosphere/*.csv")  # Update with actual pattern
file_paths = glob.glob("/home/chiaraz/thesis/36_variables/averaging/results_averaging_atmosphere/liang_res_11days_10yr_strong_avg.csv")
all_dfs = [pd.read_csv(f) for f in file_paths]

# Compute relative errors for each file
for df in all_dfs:
    df["Rel_Error_InfoFlow"] = df["Error_InfoFlow"] / df["InfoFlow"].replace([np.inf, -np.inf], np.nan)
    df["Rel_Error_Tau"] = df["Error_Tau"] / df["Tau"].replace([np.inf, -np.inf], np.nan)
    df["Rel_Error_R"] = df["Error_R"] / df["R"].replace([np.inf, -np.inf], np.nan)

# Combine data for averaging
combined_df = pd.concat(all_dfs)

import numpy as np
from scipy import stats

# Compute the median of the Tau column
mean_Tau = np.mean(combined_df["Tau"].dropna())  # Remove NaN values
print("Median Tau:", mean_Tau)

# Compute the MAD (Median Absolute Deviation) for Tau
std_Tau = np.std(combined_df["Tau"].dropna())  # Remove NaN values
print("MAD Tau:", std_Tau)

# Find outliers based on the MAD method (outliers are those > 2 MADs away from the median)
extremes_Tau = []
for i in range(len(combined_df["Tau"])):
    value = combined_df["Tau"].iloc[i]
    if not np.isnan(value):  # Skip NaNs
        # Calculate the normalized distance from the median
        normalized_distance = abs(value - mean_Tau) / std_Tau
        if normalized_distance > 3:  # If more than 2 MADs away from the median, it's an outlier
            print(f"Outlier detected: {value}, Normalized distance: {normalized_distance}")
            extremes_Tau.append(value)

print("Extreme values in Tau:", extremes_Tau)



# Create subplots for scatter plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].scatter(combined_df["Tau"], combined_df["Error_Tau"], color='b', alpha=0.4)
axs[0, 0].set_xlabel("Tau")
axs[0, 0].set_ylabel("Error Tau")
axs[0, 0].set_title("Tau vs. Error Tau")

axs[0, 1].scatter(combined_df["Tau"], combined_df["Rel_Error_Tau"], color='b', alpha=0.4)
axs[0, 1].set_xlabel("Tau")
axs[0, 1].set_ylabel("Relative Error Tau")
axs[0, 1].set_title("Tau vs. Relative Error Tau")

axs[1, 0].scatter(combined_df["InfoFlow"], combined_df["Error_InfoFlow"], color='darkorange', alpha=0.4)
axs[1, 0].set_xlabel("T")
axs[1, 0].set_ylabel("Error T")
axs[1, 0].set_title("T vs. Error T")

axs[1, 1].scatter(combined_df["InfoFlow"], combined_df["Rel_Error_InfoFlow"], color='darkorange', alpha=0.4)
axs[1, 1].set_xlabel("T")
axs[1, 1].set_ylabel("Relative Error T")
axs[1, 1].set_title("T vs. Relative Error T")

plt.tight_layout()
#plt.show()

"""
Histograms for T and tau: see the distribution of values to 
identify extremes and remove them
"""
# Remove or replace infinite values and NaNs
cleaned_info_flow = combined_df["InfoFlow"].replace([np.inf, -np.inf], np.nan).dropna()

# Check if there are any remaining NaNs or infs in cleaned_info_flow
print("Cleaned data has NaN values: ", cleaned_info_flow.isna().any())

# Now you can plot the cleaned data
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histogram for Tau
axs[0].hist(combined_df["Tau"].dropna(), bins=30, color='blue', alpha=0.6, edgecolor='black')
axs[0].set_xlabel("Tau")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Histogram of Tau Values (Averaged)")
axs[0].grid()

# Histogram for InfoFlow
axs[1].hist(cleaned_info_flow, bins=30, color='darkorange', alpha=0.6, edgecolor='black')
axs[1].set_xlabel("InfoFlow")
axs[1].set_ylabel("Frequency")
axs[1].set_title("Histogram of InfoFlow Values (Averaged)")
axs[1].grid()

plt.tight_layout()
plt.show()

## Relative error plots
# this is done to see if there is some pattern in the non significant values 
indices = np.arange(1, len(combined_df) + 1)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

axs[0].plot(indices, combined_df["Rel_Error_InfoFlow"], color='r', marker='o', linestyle='', alpha=0.6)
axs[0].set_xlabel("Index")
axs[0].set_ylabel("Relative Error InfoFlow")
axs[0].set_title("Relative Error of InfoFlow for Each Index")
axs[0].grid()

axs[1].plot(indices, combined_df["Rel_Error_Tau"], color='g', marker='o', linestyle='', alpha=0.6)
axs[1].set_xlabel("Index")
axs[1].set_ylabel("Relative Error Tau")
axs[1].set_title("Relative Error of Tau for Each Index")
axs[1].grid()

axs[2].plot(indices, combined_df["Rel_Error_R"], color='b', marker='o', linestyle='', alpha=0.6)
axs[2].set_xlabel("Index")
axs[2].set_ylabel("Relative Error R")
axs[2].set_title("Relative Error of R for Each Index")
axs[2].grid()

plt.tight_layout()
#plt.show()
