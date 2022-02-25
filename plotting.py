"""

"""

import matplotlib
from matplotlib.patches import Ellipse
import numpy as np


def init_matplotlib_params(save_not_show_fig, show_latex_fig):
    fontsize = 8
    linewidth = 1.0
    gridlinewidth = 0.7

    # Global changes
    matplotlib.rcParams.update({
            # Fonts
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize,
            'figure.titlesize': fontsize,
            # Line width
            'lines.linewidth': linewidth,
            'grid.linewidth': gridlinewidth
        })

    # Backend if saving
    if save_not_show_fig:
        matplotlib.use("pgf")

    # Font if saving or ploting in tex mode
    if save_not_show_fig or show_latex_fig:
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",   
            'font.family': 'serif',         # Use serif/main font for text elements
            'text.usetex': True,            # Use inline maths for ticks
            'pgf.rcfonts': False,           # Don't setup fonts from matplotlib rc params
        })
    return


# From https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals) # eigvals positive because covariance is positive semi definite
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)