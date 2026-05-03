import numpy as np
import matplotlib.pyplot as plt

from verifications import compare_marginal_to_single_locus_pdf  

# Parameters
deme_sizes = [10_000, 10_000]
migration_rate_values = [1e-3, 1e-4, 1e-5]
recombination_rates = [1e-4, 5e-4, 50e-4, 100e-4]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, rho in zip(axes, recombination_rates):
    for m in migration_rate_values:
        print(rho, m)
        migration_rates = np.array([
            [0, m],
            [m, 0],
        ])

        compare_marginal_to_single_locus_pdf(
            num_demes=2,
            deme_sizes=deme_sizes,
            migration_rates=migration_rates,
            recombination_rate=rho,
            max_time=200_000,
            ax=ax,
        )

    ax.set_title(f"r = {rho:.0e}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.legend()

fig.suptitle("Marginal vs Single-Locus PDF — 2 demes, N=10,000", fontsize=13)
plt.tight_layout()
plt.show()