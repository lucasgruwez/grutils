import numpy as np


def _complex_derivative(f, z, h=1e-6):
    """
    Complex-centered finite difference.

    Parameters
    ----------
    f : callable
        Complex->complex function.
    z : complex or array of complex
        Point(s) at which to evaluate the derivative.
    h : float
        Small step size.

    Returns
    -------
    df : complex or array of complex
        Approximate derivative f'(z). NaN if function is not analytic.
    """

    u = lambda z: np.real(f(z))
    v = lambda z: np.imag(f(z))

    ux = (u(z + h) - u(z - h)) / (2 * h)
    uy = (u(z + 1j * h) - u(z - 1j * h)) / (2 * h)
    vx = (v(z + h) - v(z - h)) / (2 * h)
    vy = (v(z + 1j * h) - v(z - 1j * h)) / (2 * h)

    if np.allclose(ux, vy) and np.allclose(uy, -vx):
        return ux + 1j * vx

    return np.nan  # Not analytic


def _newton_solve_point(f, w, z0, df=None, maxiter=50, tol=1e-10, damp=1.0):
    """
    Solve f(z) = w for a single complex w starting from z0 using Newton.

    Parameters
    ----------
    f : callable
        Complex->complex function.
    w : complex
        Target output value.
    z0 : complex
        Initial guess.
    df : callable or None
        Analytic derivative f'(z). If None, finite difference is used.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    damp : float
        Damping factor (0 < damp <= 1).

    Returns
    -------
    z : complex
        Approximate solution.
    """

    z = z0

    for k in range(maxiter):
        err = f(z) - w

        # Check for convergence
        if abs(err) < tol:
            return z, True, k

        dF = df(z) if df is not None else _complex_derivative(f, z)

        # Reached a singularity or non-analytic point
        if dF == 0 or np.isnan(dF) or np.isinf(dF):
            return z, False, k

        # Damped Newton step
        step = err / dF
        z = z - step * damp

        # Check for convergence based on step size
        if abs(step) < tol:
            return z, True, k + 1

    return z, False, maxiter


def numeric_inverse_on_grid(
    f,
    re_lim=(-2, 2),
    im_lim=(-2, 2),
    n_points=400,
    df=None,
    z0_seed=None,
    maxiter=50,
    tol=1e-10,
    damp=1.0,
):
    """
    Compute Z = f^{-1}(W) numerically on a mesh W over the OUTPUT plane.

    Parameters
    ----------
    f : callable
        Complex->complex function.
    re_lim, im_lim : tuple
        Output-plane bounds (Re(w), Im(w)).
    n_points : int
        Mesh resolution per axis.
    df : callable or None
        Analytic derivative f'(z). If None, finite difference is used.
    z0_seed : callable or complex or None
        Initial guess strategy. If callable, z0_seed(W) must return same-shape complex array.
        If complex, uses that constant seed. If None, seeds with W (often reasonable).
        (Warm-start continuation across the grid overrides this for interior points.)
    maxiter, tol, damp : Newton controls.

    Returns
    -------
    X, Y : 2D arrays for Re/Im of W
    Z : 2D array of complex solutions (masked where not converged or ill-conditioned)
    conv_mask : boolean mask of convergence
    """

    # Create mesh in output plane
    x = np.linspace(re_lim[0], re_lim[1], n_points)
    y = np.linspace(im_lim[0], im_lim[1], n_points)
    X, Y = np.meshgrid(x, y)
    W = X + 1j * Y

    # Initial seed array
    if callable(z0_seed):
        Z = z0_seed(W).astype(complex)
    elif z0_seed is None:
        Z = W.astype(complex)  # identity seed is a decent generic guess
    else:
        Z = np.full_like(W, complex(z0_seed))

    Z_out = np.empty_like(W, dtype=complex)  # Output solutions
    conv = np.zeros_like(W, dtype=bool)  # Convergence flags

    # Sweep with warm starts (left-to-right, top-to-bottom)
    for j in range(n_points):
        for i in range(n_points):
            # Warm start from neighbor if available (continuation)
            z0 = Z[j, i]
            if i > 0 and conv[j, i - 1]:
                z0 = Z_out[j, i - 1]
            elif j > 0 and conv[j - 1, i]:
                z0 = Z_out[j - 1, i]

            z_sol, ok, _ = _newton_solve_point(
                f, W[j, i], z0, df=df, maxiter=maxiter, tol=tol, damp=damp
            )
            Z_out[j, i] = z_sol
            conv[j, i] = ok

    # Mask points where Jacobian is tiny (ill-conditioned inverse)
    # Evaluate derivative at solutions (use FD if df not provided)
    if df is None:
        dF = _complex_derivative
        deriv = dF(f, Z_out)
    else:
        deriv = df(Z_out)
    ill = np.abs(deriv) < 1e-8
    conv_mask = conv & (~ill)

    Z_masked = np.where(conv_mask, Z_out, np.nan + 1j * np.nan)
    return X, Y, Z_masked, conv_mask


def plot_complex_map(
    ax, f, re_lim=(-2, 2), im_lim=(-2, 2), n_points=400, n_lines=11, **kwargs
):
    """
    Plot the warped grid in the OUTPUT plane by contouring Re(f^{-1}(w)) and Im(f^{-1}(w)),
    where f^{-1} is computed numerically via Newton solves.

    Colors:
      - Red contours: Re(z) = const  (images of vertical input grid lines)
      - Blue contours: Im(z) = const (images of horizontal input grid lines)

    Parameters
    ----------
    ax : matplotlib Axes
        Axes on which to plot.
    f : callable
        Complex->complex function.
    re_lim, im_lim : tuple
        Output-plane bounds (Re(w), Im(w)).
    n_points : int
        Mesh resolution per axis.
    n_lines : int
        Number of contour lines per direction.
    **kwargs : additional arguments to numeric_inverse_on_grid
    """

    # Compute numeric inverse on grid
    X, Y, Z, mask = numeric_inverse_on_grid(
        f, re_lim, im_lim, n_points=n_points, **kwargs
    )
    Zr = np.real(Z)
    Zi = np.imag(Z)

    # Choose contour levels from valid region
    valid = np.isfinite(Zr) & np.isfinite(Zi)
    if not np.any(valid):
        raise RuntimeError("Numeric inverse failed everywhere in the requested region.")

    rmin, rmax = np.nanpercentile(Zr[valid], [0, 95])
    imin, imax = np.nanpercentile(Zi[valid], [0, 95])

    r_levels = np.linspace(rmin, rmax, n_lines)
    i_levels = np.linspace(imin, imax, n_lines)

    contour_kwargs = dict(linestyles="solid", negative_linestyles="solid")
    ax.contour(X, Y, Zr, levels=r_levels, colors="red", **contour_kwargs)
    ax.contour(X, Y, Zi, levels=i_levels, colors="blue", **contour_kwargs)

    # Optional: shade non-converged/ill-conditioned areas
    if np.any(~mask):
        ax.contourf(X, Y, (~mask).astype(float), levels=[0.5, 1.5], alpha=0.08)


def plot_complex_map_forward(ax, f, re_lim=(-2, 2), im_lim=(-2, 2), over_sample=3, d_lines=1):
    """
    Plot the warped grid in the OUTPUT plane by passing a grid in the INPUT plane
    through the function f.

    Colors:
      - Red contours: Re(z) = const  (images of vertical input grid lines)
      - Blue contours: Im(z) = const (images of horizontal input grid lines)

    Parameters
    ----------
    ax : matplotlib Axes
        Axes on which to plot.
    f : callable
        Complex->complex function.
    re_lim, im_lim : tuple
        Output-plane bounds (Re(z), Im(z)).
    over_sample : int
        Oversampling factor for the input grid to improve contour quality.
    d_lines : float
        Spacing between contour lines in the input plane.
    """

    # Calculate oversampling factor
    osx = (over_sample - 1) * (re_lim[1] - re_lim[0])/2
    osy = (over_sample - 1) * (im_lim[1] - im_lim[0])/2

    re_lines = np.arange(re_lim[0] - osx, re_lim[1] + osx + d_lines, d_lines)
    im_lines = np.arange(im_lim[0] - osy, im_lim[1] + osy + d_lines, d_lines)

    for i, re in enumerate(re_lines):
        z_line = re + 1j * np.linspace(im_lim[0] - osy, im_lim[1] + osy, 1000)
        w_line = f(z_line)
        ax.plot(np.real(w_line), np.imag(w_line), color="red", lw=0.8, zorder=1)

    for i, im in enumerate(im_lines):
        z_line = np.linspace(re_lim[0] - osx, re_lim[1] + osx, 1000) + 1j * im
        w_line = f(z_line)
        ax.plot(np.real(w_line), np.imag(w_line), color="blue", lw=0.8, zorder=1)

    ax.set_xlim(re_lim)
    ax.set_ylim(im_lim)

# -------------------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example function and its derivative
    # f  = lambda z: np.log(z)
    # z0 = lambda w: np.exp(w)  # Inverse is known

    f  = lambda z: 1/z
    df = lambda z: -1/(z**2)
    z0 = lambda w: 1/w  # Inverse is known

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot_complex_map(ax, f, z0_seed=z0)
    plot_complex_map_forward(ax, f, re_lim=(-2, 2), im_lim=(-2, 2), over_sample=2, d_lines=0.2)

    ax.set_aspect("equal")
    ax.set_title(r"Complex map of $f(z)$")
    ax.set_xlabel(r"Re$(w)$")
    ax.set_ylabel(r"Im$(w)$")

    plt.show()
