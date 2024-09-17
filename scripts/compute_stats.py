"""
This module computes stats for the different feature sets in the Aim 2 paper.
"""

from scipy import stats
import numpy as np

# raw scores
f1_counts = [0.66, 0.70, 0.65, 0.65, 0.67]
f1_counts_active = [0.69, 0.73, 0.70, 0.72, 0.70]

f1_binary = [0.70, 0.73, 0.65, 0.64, 0.69]
f1_binary_active = [0.73, 0.78, 0.73, 0.69, 0.74]

f1_binary_counts = [0.69, 0.72, 0.68, 0.69, 0.68]
f1_binary_counts_active = [0.71, 0.77, 0.72, 0.69, 0.70]

# standard deviations
std_counts = [0.23, 0.14, 0.20, 0.22, 0.22]
std_counts_active = [0.20, 0.13, 0.18, 0.22, 0.20]

std_binary = [0.18, 0.15, 0.24, 0.24, 0.18]
std_binary_active = [0.18, 0.12, 0.22, 0.25, 0.17]

std_binary_counts = [0.20, 0.13, 0.18, 0.23, 0.17]
std_binary_counts_active = [0.19, 0.13, 0.16, 0.23, 0.17]

# tuples
tests = [
    (
        "counts (active vs inactive)",
        f1_counts,
        std_counts,
        f1_counts_active,
        std_counts_active,
    ),
    (
        "binary (active vs inactive)",
        f1_binary,
        std_binary,
        f1_binary_active,
        std_binary_active,
    ),
    (
        "binary_counts (active vs inactive)",
        f1_binary_counts,
        std_binary_counts,
        f1_binary_counts_active,
        std_binary_counts_active,
    ),
    ("counts vs binary", f1_counts, std_counts, f1_binary, std_binary),
    (
        "active counts vs binary counts",
        f1_counts_active,
        std_counts_active,
        f1_binary_active,
        std_binary_active,
    ),
]


def perform_t_test(inputs: tuple, n=5):

    # unpack inputs
    config, raw_f1_1, raw_std_1, raw_f1_2, raw_std_2 = inputs

    # take mean
    f11 = np.mean(raw_f1_1)
    f12 = np.mean(raw_f1_2)

    std1 = np.mean(raw_std_1)
    std2 = np.mean(raw_std_2)

    # Perform independent samples t-test
    t, p = stats.ttest_ind_from_stats(f11, std1, n, f12, std2, n)

    return config, t, p


def improvements(inputs):
    # unpack inputs
    config, raw_f1_1, raw_std_1, raw_f1_2, raw_std_2 = inputs

    # Calculate differences in mean F1-scores between Method 2 and Method 1
    differences = [raw_f1_2[i] - raw_f1_1[i] for i in range(len(raw_f1_1))]
    mean_difference = sum(differences) / len(differences)
    std_difference = (
        sum((difference - mean_difference) ** 2 for difference in differences)
        / len(differences)
    ) ** 0.5

    return config, mean_difference, std_difference


if __name__ == "__main__":
    for test in tests:
        config, t_statistic, p_val = perform_t_test(test)
        print(f"\n {config}: \n t_statistic={t_statistic}, p_val={p_val}")

    for test in tests:
        config, mean_difference, std_difference = improvements(test)
        print(
            f"\n {config}: \n mean_difference={mean_difference}, std_difference={std_difference}"
        )
