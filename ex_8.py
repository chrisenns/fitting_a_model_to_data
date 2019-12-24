# %% md
# Compute the standard uncertainty σ2 m obtained for the slope of the line found
# by the standard fit you did in Exercise 2. Now make jackknife (20 trials) and
# bootstrap estimates for the uncertainty σ2 m. How do the uncertainties compare
# and which seems most reasonable, given the data and uncertainties on the data?

# %%
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
import random

# %% Import data, do not skip any points
table_path = 'Table_1.txt'
table = pd.read_csv(table_path, sep=' ')
x, y, sy = table['x'], table['y'], table['sy']


# %%
def least_squares(x, y, sy):
    A = np.array((np.ones((len(x))), x)).T
    Y = np.asarray(y)
    sigma = np.diag(sy ** 2.)
    pars = inv(A.T @ inv(sigma) @ A) @ (A.T @ inv(sigma) @ Y)
    cov = inv(A.T @ inv(sigma) @ A)
    errs = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

    return pars, cov, errs


# %% Compute the standard uncertaines in the parameters
pars_ls, cov_ls, errs_ls = least_squares(x, y, sy)

# %% Now make 20 jackknife trials for the uncertainty in the slope
# Make measurement N times, each time leaving out data point i.
# Re-fit the data and calculate the uncertainty after each trial

N = len(x)
m_jack_vals = np.zeros(N)  # Initialize list of m_i values
for i in range(N):
    # Generate list with the i^th component removed
    x_i = np.asarray([k for j, k in enumerate(x) if j != i])
    y_i = np.asarray([k for j, k in enumerate(y) if j != i])
    sy_i = np.asarray([k for j, k in enumerate(sy) if j != i])

    # Recalculate least squares with point missing
    pars_i, cov_i, errs_i = least_squares(x_i, y_i, sy_i)

    m_jack_vals[i] = pars_i[1]  # Add to total list

m_jack = np.sum(m_jack_vals)/N

sm_jack = np.sqrt((N - 1)/N * np.sum((m_jack_vals - m_jack)**2))

# %% Now do the bootstrap method
# Select N data points WITH REPLACEMENT
random.seed(98)

indices = np.arange(N)
M = 20
m_boot_vals = np.zeros(M)  # Initialize list of m values

# Repeat for # trials
for t in range(M):

    # Choose N random index values
    rand_pts = [random.choice(indices) for _ in range(N)]

    # Get the points at the chosen indices
    x_boot = np.asarray(x[rand_pts])
    y_boot = np.asarray(y[rand_pts])
    sy_boot = np.asarray(sy[rand_pts])

    # Calculate least squares for new data set
    pars_boot, cov_boot, errs_boot = least_squares(x_boot, y_boot, sy_boot)

    m_boot_vals[t] = pars_boot[1]  # Add to total list

m = pars_ls[1]
m_boot = 1 / M * np.sum(m_boot_vals)
sm_boot = 1 / M * np.sum((m_boot_vals - m)**2)

# %%
print(r"Least squares: {:.4f} \pm {:.4f}".format(pars_ls[1], errs_ls[1]))
print(r"Jackknife: {:.4f} \pm {:.4f}".format(m_jack, sm_jack))
print(r"Bootstrap: {:.4f} \pm {:.4f}".format(m_boot, sm_boot))

# %% Given the data and their uncertainties, the jackknife estimate provides
# a more reasonable estimate of the uncertainty in the best-fit slope, in that
# it is an order of magnitude larger. The bootstrap method gives intermediate
# uncertainties, and underestimates the slope of the line.
