import numpy as np

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

__all__ = ['plot_directed']

def plot_directed(ax, x, y, *args, 
                  arrow_style='-|>', 
                  mutation_scale=15, 
                  reverse=False,
                  **kwargs):
    """
    Adds an arrow at the halfway mark of the plotted line defined by x and y.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the arrow on.
    x : array-like
        The x-coordinates of the line.
    y : array-like
        The y-coordinates of the line.
    *args : additional positional arguments
        Additional arguments passed to the plotting function.
    arrow_style : str
        The style of the arrow.
    mutation_scale : float
        The scale of the arrow.
    reverse : bool
        If True, the arrow will point in the opposite direction.
    **kwargs : keyword arguments
        Additional arguments passed to the plotting function.
    """
    
    # Make sure we have numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    if reverse:
        x, y = x[::-1], y[::-1]

    # Get z-order
    if 'zorder' in kwargs:
        zorder = kwargs.pop('zorder')
    else:
        zorder = mlines.Line2D.zorder

    l = ax.plot(x, y, *args, zorder=zorder, **kwargs)

    # Convert x-y to axes coordinates
    tx, ty = ax.transData.transform(np.vstack([x, y]).T).T
    tx, ty = ax.transAxes.inverted().transform(np.vstack([tx, ty]).T).T

    # Find midpoint
    s = np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))
    n = np.searchsorted(s, s[-1] / 2.)

    tail = np.array([x[n], y[n]])
    head = np.mean([x[n:n+2], y[n:n+2]], axis=1)

    # Add an arrow at the halfway point
    arrow = mpatches.FancyArrowPatch((tail[0], tail[1]), (head[0], head[1]),
                                    arrowstyle=arrow_style,
                                    mutation_scale=mutation_scale,
                                    zorder=zorder,
                                    color=l[0].get_color(),
                                    linewidth=l[0].get_linewidth())
    ax.add_patch(arrow)

    return l

#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # Create some hyperbolic test data
    t = np.linspace(-2, 2, 100)
    x = np.sinh(t)
    y = np.cosh(t)

    ax.set_aspect('equal')

    plot_directed(ax, x, y, color='b')
    plot_directed(ax, x, -y, color='g', reverse=True)
    plot_directed(ax, y, x, color='r')
    plot_directed(ax, -y, x, color='k', reverse=True)

    ax.grid(c='0.9')

    plt.show()
