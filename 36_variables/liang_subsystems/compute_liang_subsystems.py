"""
This file contains the function used to compute the LKIF between two subsystems 
It was donated to me by Daniel Hagan, and written based on Liang's paper of 2022 on 
complex subsystems 
"""

import numpy as np
from scipy.stats import norm

# determine if T_AB is significant 
def is_significant(TAB, error, alpha):
        """
        Check if the TAB value is significantly different from zero.

        Parameters:
            TAB (float): Computed TAB value.
            error (float): Associated standard error.
            alpha (float): Significance level.

        Returns:
            bool: True if TAB is significantly different from zero, False otherwise.
            float: Z-statistic.
            float: p-value.
        """
        # Compute Z-statistic
        Z = TAB / error

        # Two-tailed test: compute p-value
        p_value = 2 * (1 - norm.cdf(abs(Z)))

        # Compare p-value with alpha
        significant = p_value < alpha

        return significant, Z, p_value

def information_flow_subspace(xx, r, s, np_val=1, n_iter=1000, alpha=0.05):
    """
    Compute the information flow between subspaces A and B, using bootstrap resampling
    for error estimation and significance testing.

    Parameters:
        xx (ndarray): Input time series data of shape (time_points, variables).
        r (int): Index separating subspace A from the rest.
        s (int): Index separating subspace B from the rest.
        np_val (int): Time advance for Euler forward differencing.
        n_iter (int): Number of bootstrap iterations.
        conf_level (float): Confidence level for the intervals (e.g., 90, 95, 99).

    Returns:
        dict: Contains:
            - TAB (float): Information flow from A to B.
            - TBA (float): Information flow from B to A.
            - error_TAB (float): Bootstrap standard error for TAB.
            - error_TBA (float): Bootstrap standard error for TBA.
            - significant_TAB (bool): True if TAB is significant, hypothesis testing 
            - significant_TBA (bool): True if TBA is significant.
    """
    dt = 1  # Time step
    nm, M = xx.shape  # Time points (rows) and variables (columns)

    # Compute dx and truncated x
    dx1 = np.zeros((nm - np_val, M))
    for i in range(M):
        dx1[:, i] = (xx[np_val:, i] - xx[:-np_val, i]) / (np_val * dt)
    x = xx[:-np_val, :]  # Truncated dataset
    NL = nm - np_val  # Adjusted length

    # Covariance matrices
    C = np.cov(x, rowvar=False)
    Cr = C[:s, :s].copy()
    Crr = C[:s, :s].copy()

    # Modify Cr for subspace A
    Cr[:r, :] = 0
    Cr[:, :r] = 0
    np.fill_diagonal(Cr[:r, :r], 1)

    # Modify Crr for subspace B
    Crr[r:s, :] = 0
    Crr[:, r:s] = 0
    np.fill_diagonal(Crr[r:s, r:s], 1)

    # Compute dC
    dC = np.zeros((M, M))
    for i in range(M):
        for k in range(M):
            dC[k, i] = np.sum((x[:, k] - np.mean(x[:, k])) * (dx1[:, i] - np.mean(dx1[:, i]))) / (NL - 1)

    # Inverse covariance matrices
    invC = np.linalg.inv(C)
    invCr = np.linalg.inv(Cr)
    invCrr = np.linalg.inv(Crr)

    # Compute A matrix
    A = np.zeros((M, M))
    for i in range(M):
        A[:, i] = invC @ dC[:, i]
    A = A.T

    # Compute TAB and TBA
    AC = A[:s, :s] @ C[:s, :s]
    TAB = np.trace(invCr[r:s, r:s] @ AC[r:s, r:s].T) - np.trace(A[r:s, r:s])
    TBA = np.trace(invCrr[:r, :r] @ AC[:r, :r].T) - np.trace(A[:r, :r])

    # Bootstrap for error estimation and confidence intervals
    boot_TAB = np.zeros(n_iter)
    boot_TBA = np.zeros(n_iter)
    for it in range(n_iter):
        if (it % 20 == 0): 
            print("bootstrap iteration ", it)
        boot_indices = np.random.choice(NL, NL, replace=True)
        boot_x = x[boot_indices, :]
        boot_dx1 = dx1[boot_indices, :]

        # Repeat the TAB, TBA computation for bootstrapped data
        C_boot = np.cov(boot_x, rowvar=False)
        dC_boot = np.zeros((M, M))
        for i in range(M):
            for k in range(M):
                dC_boot[k, i] = np.sum(
                    (boot_x[:, k] - np.mean(boot_x[:, k])) *
                    (boot_dx1[:, i] - np.mean(boot_dx1[:, i]))
                ) / (NL - 1)

        invC_boot = np.linalg.inv(C_boot)
        A_boot = np.zeros((M, M))
        for i in range(M):
            A_boot[:, i] = invC_boot @ dC_boot[:, i]
        A_boot = A_boot.T

        AC_boot = A_boot[:s, :s] @ C_boot[:s, :s]
        boot_TAB[it] = np.trace(invCr[r:s, r:s] @ AC_boot[r:s, r:s].T) - np.trace(A_boot[r:s, r:s])
        boot_TBA[it] = np.trace(invCrr[:r, :r] @ AC_boot[:r, :r].T) - np.trace(A_boot[:r, :r])

    # Compute errors and confidence intervals
    error_TAB = np.std(boot_TAB)
    error_TBA = np.std(boot_TBA)


    return {
        "TAB": TAB,
        "TBA": TBA,
        "error_TAB": error_TAB, 
        "error_TBA": error_TBA,
        "significance_TAB (bool, Z, p-value)": is_significant(TAB, error_TAB, alpha),
        "significance_TBA (bool, Z, p-value)": is_significant(TBA, error_TBA, alpha)
    }
