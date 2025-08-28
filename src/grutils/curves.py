import numpy as np

__all__ = ['intersections']

def intersections(x1, y1, x2, y2):
    """
    Find all intersections of the curves (x1, y1) and (x2, y2).
    Based on Douglas Schwartz's algorithm.
    https://www.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections

    Parameters
    ----------
    x1, y1 : array_like
        Coordinates of the first curve.
    x2, y2 : array_like
        Coordinates of the second curve.

    Returns
    -------
    intersections : list of tuples
        A list of (x, y) coordinates where the curves intersect.

    Theory
    ------
    Given two line segments L1 and L2 with endpoints

        L1: (x1[0], y1[0]) to (x1[1], y1[1])
        L2: (x2[0], y2[0]) to (x2[1], y2[1])

    We can write the following four equations to find the intersection points

        (x1[1] - x1[0]) * t1 = x0 - x1[0]
        (y1[1] - y1[0]) * t1 = y0 - y1[0]
        (x2[1] - x2[0]) * t2 = x0 - x2[0]
        (y2[1] - y2[0]) * t2 = y0 - y2[0]

    Which in matrix form is
         _                                _   _  _     _      _ 
        | x1[1]-x1[0]       0       -1   0 | | t1 |   | -x1[0] |
        |      0       x2[1]-x2[0]  -1   0 | | t2 | = | -x2[0] |
        | y1[1]-y1[0]       0        0  -1 | | x0 |   | -y1[0] |
        |_     0       y2[1]-y2[0]   0  -1_| |_y0_|   |_-y2[0]_|

    Solving the system, if 0<=t1<=1 and 0<=t2<=1 then the line segments 
    intersect. We can cut down on the number of searches by only doing line 
    segments which have intersecting bounding boxes.
    """

    # Ensure the input arrays are numpy arrays
    x1, y1, x2, y2 = map(np.asarray, (x1, y1, x2, y2))

    # Ensure x,y pairs have the same shape
    if x1.shape != y1.shape or x2.shape != y2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Calculate number of line segments
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1

    # Find bounding boxes
    def bounding_boxes(x, y):
        x_min = np.minimum(x[:-1], x[1:])
        x_max = np.maximum(x[:-1], x[1:])
        y_min = np.minimum(y[:-1], y[1:])
        y_max = np.maximum(y[:-1], y[1:])
        return x_min, x_max, y_min, y_max

    x1_min, x1_max, y1_min, y1_max = bounding_boxes(x1, y1)
    x2_min, x2_max, y2_min, y2_max = bounding_boxes(x2, y2)

    # Indices of overlapping segments
    idx, jdx = np.where(
        (x1_max[:, None] >= x2_min[None, :]) &
        (x1_min[:, None] <= x2_max[None, :]) &
        (y1_max[:, None] >= y2_min[None, :]) &
        (y1_min[:, None] <= y2_max[None, :])
    )

    # Arrays of intersections
    x0 = []
    y0 = []

    for i, j in zip(idx, jdx):

        # Setup linear system
        A = np.array([
            [x1[i+1] - x1[i], 0, -1, 0],
            [0, x2[j+1] - x2[j], -1, 0],
            [y1[i+1] - y1[i], 0, 0, -1],
            [0, y2[j+1] - y2[j], 0, -1]
        ])
        b = -np.array([x1[i], x2[j], y1[i], y2[j]])

        # Solve linear system
        t = np.linalg.solve(A, b)

        # Check if solution is valid
        if 0 <= t[0] <= 1 and 0 <= t[1] <= 1:
            x0.append(x1[i] + t[0] * (x1[i + 1] - x1[i]))
            y0.append(y1[i] + t[0] * (y1[i + 1] - y1[i]))

    return list(zip(x0, y0))

#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    x1 = np.linspace(0, 1, 100)
    y1 = np.sin(2 * np.pi * x1)

    x2 = np.linspace(0, 1, 100)
    y2 = np.cos(2 * np.pi * x2)

    plt.plot(x1, y1, label="sin")
    plt.plot(x2, y2, label="cos")

    inter = intersections(x1, y1, x2, y2)
    for x, y in inter:
        plt.plot(x, y, 'ro')

    plt.legend()
    plt.show()