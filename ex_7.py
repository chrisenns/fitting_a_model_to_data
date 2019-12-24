# %% md Exercise 7: Solve Exercise 6 but now plot the fully marginalized (
# over m, b, Yb, Vb) posterior distribution function for parameter Pb. Is
# this distri- bution peaked about where you would expect, given the data?
# Now repeat the problem, but dividing all the data uncertainty variances
# σ^2_yi by 4 (or dividing the uncertainties σ_yi by 2). Again plot the fully
# marginalized posterior distribution function for parameter Pb. Your plots
# should look something like those in Figure 5. Discuss.

# %%
import corner
from IPython import display
import emcee
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


# %% Import data, do not skip any points
table_path = 'Table_1.txt'
table = pd.read_csv(table_path, sep=' ')
x, y, sy = table['x'], table['y'], table['sy']


# %% Define the probabilistic model...

labels = ["$m$", "$b$", "$P$", "$Y$", "$ln V$"]
bounds = [(1.4, 3), (-20, 100), (0, 1), (100, 700), (4, 16)]

# A simple prior:
def lnprior(p):
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0


# The "foreground" linear likelihood:
def lnlike_fg(p):
    m, b, _, Y, lnV = p
    model = m * x + b
    return -0.5 * (((model - y) / sy) ** 2 + 2 * np.log(sy))


# The "background" outlier likelihood:
def lnlike_bg(p):
    _, _, P, Y, lnV = p
    var = np.exp(lnV) + sy**2
    return -0.5 * ((Y - y) ** 2 / var + np.log(var))


# Full probabilistic model.
def lnprob(p):
    m, b, P, Y, lnV = p

    # First check the prior.
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf, None

    # Compute the vector of foreground likelihoods and include the q prior.
    ll_fg = lnlike_fg(p)
    arg1 = ll_fg + np.log(1.0 - P)

    # Compute the vector of background likelihoods and include the q prior.
    ll_bg = lnlike_bg(p)
    arg2 = ll_bg + np.log(P)

    # Combine these using log-add-exp for numerical stability.
    ll = np.sum(np.logaddexp(arg1, arg2))

    # We're using emcee's "blobs" feature in order to keep track of the
    # foreground and background likelihoods for reasons that will become
    # clear soon.
    return lp + ll, (arg1, arg2)


def run_walkers(p0, ndim, nwalkers):
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 500);

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1000);

    return sampler.flatchain


# %% Initialize the walkers at a reasonable location.
ndim, nwalkers = 5, 32
p0 = np.array([2, 0, 0.5, 200, 10])
samples = run_walkers(p0, ndim, nwalkers)

# %% Show corner plot
corner.corner(samples, bins=35, range=bounds, labels=labels)
plt.show()

# %% Recreate that particular plot from the corner plot
samples_df = pd.DataFrame(samples, columns=labels)

plt.hist(samples_df["$P$"], bins=32)
plt.show()

# %% The marginal distribution for $P_b$ is left-skewed and peaked at about 0.25
# Given the data, I suppose this seems reasonable.

# %%
# Now divide the errors in half and redo the analysis
sy = table['sy']/2

# %%
samples = run_walkers(p0, ndim, nwalkers)

# %%
samples_df_half = pd.DataFrame(samples, columns=labels)

plt.hist(samples_df_half["$P$"], bins=32)
plt.show()

# %% md
# When the uncertainties are smaller, the marginal distribution is more
# symmetric and centred at a higher value. The smaller uncertainties mean the
# inference is more likely to detect a given point as an outlier in this model.
