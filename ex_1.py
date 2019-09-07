#%% md
# Exercise 1: Using the standard linear algebra method of this Section, fit the straight line y = mx + b to the x, y,
# and σy values for data points 5 through 20 in Table 1 on page 6. That is, ignore the first four data points, and also
# ignore the columns for σx and ρxy. Make a plot showing the points, their uncertainties, and the best-fit line. Your
# plot should end up looking like Figure 1. What is the standard uncertainty variance σ2 m on the slope of the line?

#%%
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
# Import data
table_1 = 'Table_1.txt'
with open(table_1) as f:
    ncols_1 = len(f.readline().split(','))
ID, x, y, sy, sx, pxy = np.loadtxt(table_1, skiprows=5, usecols=range(2, ncols_1), unpack=True, delimiter=' ')

# Skipped first 4 data points

# %%
# Create A with first column 1's, second column x's
# This makes Y = AX yield y = m*x + b for each data pair (y, x), and X=[m,b]
A = np.array((np.ones((len(x))), x)).T

# Create C with diagonal the (sy^2)'s
C = np.diag(sy**2.)

# Compute X, with X[0] = b, X[1] = m
X = inv(A.T @ inv(C) @ A) @ (A.T @ inv(C) @ y)

# Compute the covariant matrix for X
cov_X = inv(A.T @ inv(C) @ A)

# The standard uncertainty variance on the slope of the line is
np.sqrt(cov_X[1, 1])

# Make model y = mx + b
t = np.linspace(min(x), max(x), 100)
Y = X[1]*t + X[0]

# %%
# Plot results
fig, ax = plt.subplots(1)
datastyle = dict(linestyle='none', marker='o', color='k', ecolor='#666666')
ax.errorbar(x, y, yerr=sy, **datastyle)
ax.plot(t, Y)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
text = r"$y = ({:.2f} \pm {:.2f}) x + ({:.0f} \pm {:.0f})$"
ax.text(100, 160, text.format(X[1], np.sqrt(cov_X[1, 1]), X[0], np.sqrt(cov_X[0, 0])))
plt.show()
