import numpy as np
from scipy.stats import gamma
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from pathlib import Path

# Create some arrarys from the pdf
x = np.linspace(0,2000, 1000)
y = gamma.pdf(x, 5, scale=150)
y2 = gamma.pdf(x, 3.5, scale=100)
y3 = gamma.pdf(x, 2.5, scale=75)

# plot arrays using matplotlib
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(x, y, label=r'$k=5, \theta=150$')
ax.plot(x, y2, label=r'$k=3.5, \theta=100$')
ax.plot(x, y3, label=r'$k=2.5, \theta=75$')
ax.set_xlabel(r'x')
ax.set_ylabel(r'$f(x)$')
ax.set_title(r'$f(x)=\frac{1}{\Gamma(k)\theta^k}x^{k-1}e^{-\frac{x}{\theta}}$')
ax.legend(loc='best', frameon=False)
fig.savefig(Path().joinpath("Plots", "GammaDist.png"))