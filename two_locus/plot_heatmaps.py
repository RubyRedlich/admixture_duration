from markov_model import *
import matplotlib.pyplot as plt
from verifications import *

# Plot the joint distribution for different recombination rates and distances
NUM_DEMES = 2
DEME_SIZES = [10_000, 10_000]
migration_rates = [1e-5, 1e-4, 1e-3]
recombination_rates = [0, 1e-4, 5e-4, 50e-4]
s0 = ((1,1),(1,1))

nrow = len(migration_rates)
ncol = len(recombination_rates)
fig, axes = plt.subplots(nrow,ncol,figsize=(4*ncol, 3*nrow))

# Calculate all pdfs
pdfs = []
for i in range(len(migration_rates)):
    for j in range(len(recombination_rates)):
        print(i,j)
        mrates = np.fliplr(np.eye(NUM_DEMES)) * migration_rates[i]
        model = TwoLocusMarkovModel(NUM_DEMES, mrates, DEME_SIZES, recombination_rates[j])
        model.eval_joint_pdf(s0, max_time=20_000, nintervals=500)
        pdf, time_bins = model.joint_pdf
        pdfs.append((i, j, pdf, time_bins))
        print(model.get_marginal_pdf().sum()) # check dist sums to ~1

# Find min/max for shared color bar
vmin = min(pdf.min() for _, _, pdf, _ in pdfs)
vmax = max(pdf.max() for _, _, pdf, _ in pdfs)

# Plot all pdfs
for i, j, pdf, time_bins in pdfs:
    ax = axes[i][j]
    im = ax.pcolormesh(time_bins, time_bins, pdf, cmap='Blues', vmin=vmin, vmax=vmax)
    ax.set_title(f'm={migration_rates[i]:.0e}, r={recombination_rates[j]:.0e}', fontsize=8)
    ax.set_xlabel('T1', fontsize=7)
    ax.set_ylabel('T2', fontsize=7)
    ax.tick_params(labelsize=6)

# Single shared colorbar
fig.colorbar(im, ax=axes, label='PDF', shrink=0.6)

plt.suptitle('Joint PDF by Migration and Recombination Rate', fontsize=12)
plt.show()

# # Also compare the marginal distributions
# for i, j, pdf, time_bins in pdfs:
#     mpdf = marginal_pdf(pdf, time_bins)
#     plt.plot(
#         time_bins, mpdf, 
#         label=f'm={migration_rates[i]:.0e}, r={recombination_rates[j]:.0e}'
#     )
# plt.legend()
# plt.show()

