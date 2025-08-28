import numpy as np

__all__ = ["grad_mat", "cumtrapz_mat", "trapz_mat"]

def grad_mat(t):
    """
    Gradient matrix. Returns a matrix A such that Ax = dx/dt.

    Parameters
    ----------
    t : array_like
        Time vector.

    Returns
    -------
    A : ndarray
        Gradient matrix.
    """

    # Using centred differences, except at the boundaries where we use
    # forward/backward differences
    t  = np.asarray(t, dtype=float)
    dt = np.diff(t)

    A = np.zeros((t.size, t.size), dtype=float)
    for i in range(t.size):
        # Treat boundary conditions
        if i == 0:
            A[i,i]   = -1/dt[i]
            A[i,i+1] =  1/dt[i]
        elif i == t.size - 1:
            A[i,i-1] = -1/dt[i-1]
            A[i,i]   =  1/dt[i-1]
        else:
            A[i,i-1] = -1/(dt[i-1] + dt[i])
            A[i,i+1] =  1/(dt[i-1] + dt[i])

    return A

def cumtrapz_mat(t):
    """
    Trapezoidal integration matrix. Returns a matrix A such that
    Ax = ∫x dt.

    Parameters
    ----------
    t : array_like
        Time vector.

    Returns
    -------
    A : ndarray
        Integration matrix.
    """

    t = np.asarray(t, dtype=float)
    dt = np.diff(t)

    A = np.zeros((t.size, t.size), dtype=float)

    for i in range(1, t.size):
        A[i:,i-1] += dt[i-1]/2
        A[i:,i]   += dt[i-1]/2

    return A

def trapz_mat(t):
    """
    Trapezoidal integration matrix. Returns a matrix A such that
    Ax = ∫_0^t x dt.

    Parameters
    ----------
    t : array_like
        Time vector.

    Returns
    -------
    A : ndarray
        Integration matrix.
    """

    t = np.asarray(t, dtype=float)
    dt = np.diff(t)

    A = np.zeros(t.size, dtype=float)

    for i in range(1, t.size):
        A[i-1] += dt[i-1]/2
        A[i]   += dt[i-1]/2

    return A

#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy import integrate

    # Non-uniform grid
    x = np.linspace(0, 1, 100)**2
    y = np.sin(2 * np.pi * x)

    dy = np.cos(2 * np.pi * x) * 2 * np.pi
    Iy = integrate.cumulative_trapezoid(y, x, initial=0)

    A = grad_mat(x)
    B = cumtrapz_mat(x)

    print(integrate.trapezoid(y, x))
    print(trapz_mat(x) @ y)

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(x, y, label="y")

    ax[1].plot(x, dy, label="dy")
    ax[1].plot(x, A @ y, "--", label="Ay")

    ax[2].plot(x, Iy, label="Iy")
    ax[2].plot(x, B @ y, "--", label="By")

    for a in ax: 
        a.legend()
        a.grid(color='0.8')

    plt.show()