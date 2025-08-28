import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Matplotlib default styling
#-------------------------------------------------------------------------------

# Setup matplotlib for latex compatible figures
def pyplot_latex():
    plt.rcParams['text.usetex']      = True
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family']      = 'serif'

# set fontsize
plt.rcParams['font.size']        = 12

# Set linewidth
plt.rcParams['axes.linewidth']   = 0.8
plt.rcParams['grid.linewidth']   = 0.8
plt.rcParams['lines.linewidth']  = 1.2
plt.rcParams["patch.linewidth"]  = 0.8

# Set grid
plt.rcParams['grid.color']      = '0.9'

# Ticks on inside both sides of the plot
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['xtick.top']        = True
plt.rcParams['ytick.right']      = True

# Matlab-esque legend
def matlab_legend():
    plt.rcParams['legend.frameon']   = True
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox']  = False

#-------------------------------------------------------------------------------
# Import utilities
#-------------------------------------------------------------------------------

from . import plotting
from . import calculus
from . import curves

__all__ = [
    # Matplotlibrc functions
    'pyplot_latex', 'matlab_legend',
    # Modules
    'plotting', 'calculus', 'curves'
]