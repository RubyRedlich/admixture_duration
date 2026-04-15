import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

def hellinger_normal(mu1, sigma1, mu2, sigma2):
    # https://www.stat.cmu.edu/~larry/=stat705/Lecture27.pdf MULTIPLY BY 2 TO BE CONSISTENT WITH DEFN IN THESE LECTURE NOTES 
    # most of the time H2 is multiplied by 1/2
    return (1 - np.sqrt(2*sigma1*sigma2/(sigma1**2 + sigma2**2))*np.exp(-0.25*(mu1-mu2)**2/(sigma1**2+sigma2**2))) * 2

def TVD_normal(mu1, sigma1, mu2, sigma2):
    assert sigam1 == sigma2 # TVD is defined as such for gaussians with different means but same variances
    z = np.abs(mu1-mu2) / (2*sigma1)
    return 2*norm.cdf(z)

def sample_normal(mu, sigma, ntrials=100, n=50):
    return np.random.normal(mu, sigma, size=(ntrials, n))

def loglik_normal(X, mu, sigma):
    return norm.logpdf(X, loc=mu, scale=sigma).sum(axis=1)

def bayes_error(params, ntrials, n, loglik_fn=loglik_normal, sample_fn=sample_normal, H2_fn=hellinger_normal):
    # Sample
    X_P = sample_fn(ntrials=ntrials, n=n, **params["P"])
    X_Q = sample_fn(ntrials=ntrials, n=n, **params["Q"])

    # LRT
    lambda_P = loglik_fn(X_P, **params["P"]) - loglik_fn(X_P, **params["Q"])
    lambda_Q = loglik_fn(X_Q, **params["P"]) - loglik_fn(X_Q, **params["Q"])

    # Bayes error
    alpha = (lambda_P < 0).sum() /  ntrials
    beta = (lambda_Q > 0).sum() / ntrials
    bayes_error = (alpha + beta) / 2

    # Hellinger bound
    all_params = {k+'1':v for k,v in params["P"].items()} | {k+'2':v for k,v in params["Q"].items()}
    H2 = H2_fn(**all_params)
    bayes_error_upperbound = np.exp(-n*H2/2) # H2/2 is because our definition of H2 does not include the factor of 1/2
    H2n = 2 * (1 - (1-H2/2)**n) 
    TVD_bounds = np.array([H2n/2, np.sqrt(H2n)])
    TVD_bounds = 0.5 - TVD_bounds/2

    return bayes_error, bayes_error_upperbound, TVD_bounds[0], TVD_bounds[1]

def print_output(bayes_error, bayes_error_upperbound, TVD_bounds):
    output = f"The empirical bayes error estimate: {bayes_error}\n"
    output += f"The bayes error upper bound: {bayes_error_upperbound}\n"
    output += f"The upper and lower bounds on the TVD estimate of bayes error: [{TVD_bounds[1]},{TVD_bounds[0]}]"
    print(output)

# Plotting function from Claude
def plot_bayes_error(n_values, empirical, bayes_upper, tvd_lower, tvd_upper,
                     figsize=(8, 5), title="Bayes Error and Bounds vs. Number of Samples"):
    """
    Plot empirical Bayes error alongside upper bound and TVD-based bounds.
 
    Parameters
    ----------
    n_values    : array-like, values of n (x-axis)
    empirical   : array-like, empirical Bayes error
    bayes_upper : array-like, upper bound on Bayes error
    tvd_lower   : array-like, lower bound from TVD:  (1 - TVD) / 2
    tvd_upper   : array-like, upper bound from TVD:  (1 - TVD) / 2  [or 1/2 * exp(-nD)]
    """

    # LOWER BOUND TVD_LOWER AT 0
    tvd_lower[tvd_lower < 0] = 0

    fig, ax = plt.subplots(figsize=figsize)
 
    # --- shaded TVD band ---
    ax.fill_between(n_values, tvd_lower, tvd_upper,
                    alpha=0.15, color="#2196F3", label="TVD bound region")
 
    # --- TVD bound lines ---
    ax.plot(n_values, tvd_lower, color="#2196F3", linewidth=1.5,
            linestyle="--", label="TVD lower bound")
    ax.plot(n_values, tvd_upper, color="#2196F3", linewidth=1.5,
            linestyle=":",  label="TVD upper bound")
 
    # --- Bayes error upper bound ---
    ax.plot(n_values, bayes_upper, color="#FF5722", linewidth=1.5,
            linestyle="-.", label="Bayes error upper bound")
 
    # --- Empirical Bayes error (most prominent) ---
    ax.plot(n_values, empirical, color="#212121", linewidth=2.5,
            linestyle="-", label="Empirical Bayes error", zorder=5)
 
    # --- reference line at 0.5 ---
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(n_values[-1], 0.505, "random guess (0.5)",
            va="bottom", ha="right", fontsize=8, color="gray")
 
    # --- formatting ---
    ax.set_xlabel("Number of samples $n$", fontsize=12)
    ax.set_ylabel("Bayes error probability", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    # ax.set_ylim(0, 0.55)
    ax.set_xlim(n_values[0], n_values[-1])
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
 
    return fig, ax

# # PLOT THE RELATIONSHIP BETWEEN THE BOUNDS, ERROR, AND 
# params ={
#     "P":{"mu":0,"sigma":1},
#     "Q":{"mu":0.1,"sigma":1}
# }
# n_vals = [n for n in range(1,5000,25)]
# outputs = np.array([list(bayes_error(params, ntrials=100, n=n)) for n in n_vals]) # bayes_error, upper bound, TVD_upper, TVD_lower
# fig, ax = plot_bayes_error(n_vals, outputs[:,0], outputs[:,1], outputs[:,3], outputs[:,2])
# plt.show()









