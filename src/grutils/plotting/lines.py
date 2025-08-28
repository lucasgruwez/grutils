import numpy as np

import matplotlib.collections as mcoll

__all__ = ['plot_variable_width', 'plot_variable_color']

def plot_variable_width(ax, x, y, widths, **kwargs):
    """
    Plot a line with variable width.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x, y : array-like
        Coordinates of the line.
    widths : array-like
        Widths at each segment (length = len(x)-1)
    kwargs : additional keyword arguments passed to LineCollection (e.g., color)
    """
    
    # Create segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = mcoll.LineCollection(segments, linewidths=widths, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    return ax

def plot_variable_color(ax, x, y, colors, cmap='viridis', **kwargs):
    """
    Plot a line with variable color.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x, y : array-like
        Coordinates of the line.
    colors : array-like
        Color values for each segment (length = len(x)-1)
    cmap : str or Colormap
        Colormap to use for mapping colors
    kwargs : additional keyword arguments passed to LineCollection
    """
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = mcoll.LineCollection(segments, array=np.array(colors), cmap=cmap, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    return ax


#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    widths = np.linspace(0.5, 5, len(x)-1)
    colors = np.linspace(0, 1, len(x)-1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_variable_width(ax[0], x, y, widths, color='blue')
    ax[0].set_title("Variable Width")

    plot_variable_color(ax[1], x, y, colors, cmap='plasma')
    ax[1].set_title("Variable Color")

    plt.show()
