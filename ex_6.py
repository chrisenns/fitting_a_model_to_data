#%% md
# Exercise 6: Using the mixture model proposed above—that treats the distribution as a mixture of a thin line containing
# a fraction [1−Pb] of the points and a broader Gaussian containing a fraction Pb of the points—find the best- fit
# (the maximum a posteriori) straight line y = mx + b for the x, y, and σy for the data in Table 1 on page 6. Before
# choosing the MAP line, marginalize over parameters (Pb, Yb, Vb). That is, if you take a sampling approach, this means
# sampling the full five-dimensional parameter space but then choosing the peak value in the histogram of samples in the
# two-dimensional parame- ter space (m, b). Make one plot showing this two-dimensional histogram, and another showing
# the points, their uncertainties, and the MAP line. How does this compare to the standard result you obtained in
# Exercise 2? Do you like the MAP line better or worse? For extra credit, plot a sampling of 10 lines drawn from the
# marginalized posterior distribution for (m, b) (marginalized over Pb, Yb, Vb) and plot the samples as a set of light
# grey or transparent lines. Your plot should look like Figure 4.

#%%
import corner
from IPython import display
import emcee
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


#%% Import data, do not skip any points
table_path = 'Table_1.txt'
table = pd.read_csv(table_path, sep=' ')
x, y, sy = table['x'], table['y'], table['sy']

# Make another set with outliers trimmed by hand
table_trim = table[4:]
x_trim, y_trim, sy_trim = table_trim['x'], table_trim['y'], table_trim['sy']


#%% md
# First, redo the least-squares approach to fitting a line by making the model an object with the data as attributes.


#%%
class StraightLineLstSqModel:
    def __init__(self, u, v, sv):
        """
        Store data as object attribute for ease of access.
        :param u: list, array;
            set of x-values of data
        :param v: list, array;
            set of y-values of data
        :param sv: list, array;
            set of uncertainties sigma_y in y corresponding to each data point
        :return: None
        """
        self.x = np.asarray(u)
        self.y = np.asarray(v)
        self.sy = np.asarray(sv)

    def line_model(self, pars):
        """
        Evaluate a straight ine model at given x values.
        :param pars: tuple, list, array;
            b, m = intercept, slope of line
        :return: y = b + m*x
        """
        b, m = pars
        return b + m*self.x

    def least_squares(self):
        """
        Find the optimal straight-line model to the data using least-squares.
        :return: pars: array, cov: array;
            best-fit pars = (intercept, slope) and covariance matrix
        """
        from numpy.linalg import inv

        u = np.array((np.ones((len(self.x))), self.x)).T
        v = np.asarray(self.y)
        sigma = np.diag(self.sy ** 2.)
        pars = inv(u.T @ inv(sigma) @ u) @ (u.T @ inv(sigma) @ v)
        cov = inv(u.T @ inv(sigma) @ u)

        return pars, cov

    def plot_lstsq_model(self):
        """
        Plot data with error bars and linear least-squares fit line.
        :return: None
        """
        pars1, cov1 = self.least_squares()
        b1, m1 = pars1
        b_err1, m_err1 = np.sqrt(cov1[0, 0]), np.sqrt(cov1[1, 1])

        x1 = np.linspace(min(self.x), max(self.x), 100)
        y1 = m1 * x1 + b1

        fig1, ax1 = plt.subplots(1, 1)
        data_style = dict(linestyle='none', marker='o', color='k', ecolor='#666666')
        ax1.errorbar(self.x, self.y, yerr=self.sy, **data_style)
        ax1.plot(x1, y1, color='b')
        ax1.set_xlabel(r'$x$', fontsize=20)
        ax1.set_ylabel(r'$y$', fontsize=20)

        text = r'$y = ({:.0f} \pm {:.0f}) + ({:.2f} \pm {:.2f}) x$'
        xloc1 = 100
        yloc1 = 175
        ax1.text(x=xloc1, y=yloc1, s=text.format(b1, b_err1, m1, m_err1))

        ax1.autoscale()
        fig1.tight_layout()
        plt.show()


#%%
lstsq_model = StraightLineLstSqModel(x, y, sy)
lstsq_model.plot_lstsq_model()

pars_ls, cov_ls = lstsq_model.least_squares()
b_ls, m_ls = pars_ls
b_ls_err, m_ls_err = np.sqrt(cov_ls[0, 0]), np.sqrt(cov_ls[1, 1])
x0 = np.linspace(min(x), max(x), 200)
y0 = m_ls * x0 + b_ls

#%%
lstsq_model_trim = StraightLineLstSqModel(x_trim, y_trim, sy_trim)
lstsq_model_trim.plot_lstsq_model()

pars_ls_trim, cov_ls_trim = lstsq_model_trim.least_squares()
b_ls_trim, m_ls_trim = pars_ls_trim
b_ls_err_trim, m_ls_err_trim = np.sqrt(cov_ls_trim[0, 0]), np.sqrt(cov_ls_trim[1, 1])
x0_trim = np.linspace(min(x), max(x), 200)
y0_trim = m_ls_trim * x0 + b_ls_trim


#%% md
# Find best fit using a Bayesian model. Should give identical result.
# Comes from https://dfm.io/posts/mixture-models/


#%%
# Define the probabilistic model...
# A simple prior:
labels = ["$m$", "$b$", "$P$", "$Y$", "$\ln V$"]
bounds = [(1, 3), (-20, 100), (0, 1), (100, 600), (0, 20)]
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


#%%
%%time
# Initialize the walkers at a reasonable location.
ndim, nwalkers = 5, 32
p0 = np.array([2, 0, 0.5, 200, 10])
p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]

# Set up the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Run a burn-in chain and save the final location.
pos, _, _, _ = sampler.run_mcmc(p0, 500);

# Run the production chain.
sampler.reset()
sampler.run_mcmc(pos, 1000);


#%%
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.flatchain
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, i], "k", alpha=0.5)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()

#%%
labels = ["$m$", "$b$", "$P$", "$Y$", "$\ln V$"]
corner.corner(samples, bins=35, range=bounds, labels=labels)
plt.show()


#%%
low, med, hi = np.percentile(samples, [16, 50, 84], axis=0)
upper, lower = hi - med, med - low

disp_str = ""
for i, name in enumerate(labels):
    fmt_str = r'{name}={val:.2f}^{{+{plus:.2f}}}_{{-{minus:.2f}}}'
    disp_str += fmt_str.format(name=name, val=med[i], plus=upper[i],
                               minus=lower[i])
    disp_str += r'\quad '

disp_str = "${}$".format(disp_str)
display.Latex(data=disp_str)


#%%
norm = 0.0
post_prob = np.zeros(len(x))
for i in range(sampler.chain.shape[1]):
    for j in range(sampler.chain.shape[0]):
        ll_fg, ll_bg = sampler.blobs[i][j]
        post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        norm += 1
post_prob /= norm
print(", ".join(map("{0:.3f}".format, post_prob)))


#%%
# Plot the prediction.
fig, ax = plt.subplots(1, figsize=(15,10))

# Compute the quantiles of the predicted line and plot them.
A = np.vander(x0, 2)
lines = np.dot(sampler.flatchain[:, :2], A.T)
quantiles = np.percentile(lines, [16, 50, 84], axis=0)

# ax.fill_between(x0, quantiles[0], quantiles[2], color="#8d44ad", alpha=0.5,
#                 label='Range of mixture models between quartiles')

n = 10  # # lines to plot
for line in lines[::int(len(lines)/10)]:
    plt.plot(x0, line, color='b', alpha=0.4)

ax.plot(x0, quantiles[1], color="r", label='Median mixture model')

# Plot the data points.
ax.errorbar(x, y, yerr=sy, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
# Plot the (true) outliers.
pcm = ax.scatter(x, y, marker="s", s=100, c=post_prob, cmap="gray_r", 
                 norm=LogNorm(vmin=.50, vmax=1), zorder=1000)

# Show colorbar describing outlier probabilities
cb = fig.colorbar(pcm, ax=ax)
cb.set_label('Probability of not being an outlier', fontsize=20)
cb.ax.tick_params(labelsize=18)

ls_text = r'Least squares: $y = ({:.2f} \pm {:.2f}) x + ({:.0f} \pm {:.0f})$'
xloc = 90
yloc = 175
ax.text(x=xloc, y=yloc, s=ls_text.format(m_ls, m_ls_err, b_ls, b_ls_err), fontsize=20)

mcmc_text = r'Mixture model: $y = ' + \
            r'({val_m:.2f}^{{+{plus_m:.2f}}}_{{-{minus_m:.2f}}})x$ +' + \
            r'$({val_b:.2f}^{{+{plus_b:.2f}}}_{{-{minus_b:.2f}}})$'
xloc = 90
yloc = 140
ax.text(x=xloc, y=yloc, s=mcmc_text.format(val_m=med[0], plus_m=upper[0], minus_m=lower[0], 
                                           val_b=med[1], plus_b=upper[1], minus_b=lower[1]), 
                                           fontsize=20)
ax.tick_params('both', labelsize=18)
ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$y$', fontsize=20)

ax.legend(fontsize=18)
plt.show()

#%%
