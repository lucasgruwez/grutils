import numpy as np

from matplotlib.patches import PathPatch
from matplotlib.path import Path

__all__ = ['plot_error_band']

def plot_error_band(ax, x, y, err, **kwargs):
    """
    Plot an error band around a parametric curve. Taken from the matplotlib
    gallery.

    https://matplotlib.org/stable/gallery/lines_bars_and_markers/curve_error_band.html

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the error band on.
    x : array-like
        The x-coordinates of the curve.
    y : array-like
        The y-coordinates of the curve.
    err : array-like
        The error values (positive and negative) to use for the band.
    **kwargs : keyword arguments
        Additional arguments passed to the patch artist.

    """
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))
    
#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a cardioid curve
    N = 400
    t = np.linspace(0, 2 * np.pi, N)
    r = 0.5 + np.cos(t)
    x, y = r * np.cos(t), r * np.sin(t)

    _, axs = plt.subplots(1, 2, layout='constrained', sharex=True, sharey=True)

    # Plot settings for loop
    errs = [
        (axs[0], "constant error", 0.05),
        (axs[1], "variable error", 0.05 * np.sin(2 * t) ** 2 + 0.04),
    ]

    for i, (ax, title, err) in enumerate(errs):
        ax.set(title=title, aspect=1, xticks=[], yticks=[])
        ax.plot(x, y, "k")
        plot_error_band(ax, x, y, err=err,
                        facecolor=f"C{i}", edgecolor="none", alpha=.3)

    plt.show()