#%% md
# Exercise 3: Generalize the method of this Section to fit a general quadratic (second order relationship. Add another
# column to matrix A containing the values x2 i , and another element to vector X (call it q). Then re-do Exercise 1 but
# fitting for and plotting the best quadratic relationship g(x) = q x2 +mx + b. Your plot should end up looking like
# Figure 3.

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
# Create A with first column 1's, second column x's, third column x^2's
# This makes Y = AX yield y = q*x^2 + m*x + b for each data pair (y, x), and X=[q,m,b]
A = np.array((np.ones((len(x))), x, x**2.)).T

# Create C with diagonal the (sy^2)'s
C = np.diag(sy**2.)

# Compute X, with X[0] = b, X[1] = m, X[2] = q
X = inv(A.T @ inv(C) @ A) @ (A.T @ inv(C) @ y)

# Compute the covariant matrix for X: cov(X) = (AT*Cinv*A)inv
cov_X = inv(A.T @ inv(C) @ A)

# Make model y=mx+b
t = np.linspace(min(x), max(x), 100)
Y = X[2]*t**2. + X[1]*t + X[0]

# %%
fig, ax = plt.subplots(1)
datastyle = dict(linestyle='none', marker='o', color='k', ecolor='#666666')
ax.errorbar(x, y, yerr=sy, **datastyle)
ax.plot(t, Y)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
text = r"$y = ({:.4f}\pm{:.4f}) x^2 + ({:.2f}\pm{:.2f}) x + ({:.0f}\pm{:.0f})$"
ax.text(80, 160, text.format(X[2], np.sqrt(cov_X[2, 2]), X[1], np.sqrt(cov_X[1, 1]), X[0], np.sqrt(cov_X[0, 0])))

plt.show()
