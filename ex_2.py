#%% md
# Exercise 2: Repeat Exercise 1 but for all the data points in Table 1 on page 6. Your plot should end up looking like
# Figure 2. What is the standard uncertainty variance σ2 m on the slope of the line? Is there anything you don’t like
# about the result? Is there anything different about the new points you have included beyond those used in Exercise 1?

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
ID, x, y, sy, sx, pxy = np.loadtxt(table_1, skiprows=1, usecols=range(2, ncols_1), unpack=True, delimiter=' ')

# Do not skip any data

# %%
# Create A with first column 1's, second column x's
# This makes Y = AX yield y = m*x + b for each data pair (y, x), and X=[m,b]
A = np.array((np.ones((len(x))), x)).T

# Create C with diagonal the (sy^2)'s
C = np.diag(sy**2.)

# Compute X, with X[0] = b, X[1] = m
X = inv(A.T @ inv(C) @ A) @ (A.T @ inv(C) @ y)

# Compute the covariant matrix for X: cov(X) = (AT*Cinv*A)inv
cov_X = inv(A.T @ inv(C) @ A)

# Make model y=mx+b
t = np.linspace(min(x), max(x), 100)
Y = X[1]*t + X[0]

# %%
fig, ax = plt.subplots(1)
datastyle = dict(linestyle='none', marker='o', color='k', ecolor='#666666')
ax.errorbar(x, y, yerr=sy, **datastyle)
ax.plot(t, Y)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
text = r"$y = ({:.2f} \pm {:.2f}) x + ({:.0f} \pm {:.0f})$"
ax.text(100, 160, text.format(X[1], np.sqrt(cov_X[1, 1]), X[0], np.sqrt(cov_X[0, 0])))
plt.show()

#%%
# From the plot, the slope of the fit line does not appear to follow the apparent slope of the bulk of the data points.
# However, the standard errors in both the slope and the intercept are lower than when the outliers are excluded, which
# does not match our intuition about the data.
