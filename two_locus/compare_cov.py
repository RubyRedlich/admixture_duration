import numpy as np
import matplotlib.pyplot as plt
from verifications import *  

# Parameters
# deme_sizes = [10_000, 10_000]
# migration_rate_values = [1e-3, 1e-4, 1e-5]
recombination_rates = [0.5e-4, 1e-4, 5e-4, 10e-4, 50e-4, 100e-4]

fig, axes = plt.subplots(1, 1, figsize=(4, 4))
axes = [axes]

for ax in axes:
    compare_covariance(
        1, [5_000], np.array([0]), recombination_rates=recombination_rates,
        nintervals=500, min_time=1, max_time=80_000,
        ax=ax, s0_two_locus = ((1,1),(1,1))
    )
    ax.set_xlabel("Rho")
    ax.set_ylabel("Covariance")
    ax.legend()

fig.suptitle("Covariance", fontsize=13)
plt.tight_layout()
plt.show()