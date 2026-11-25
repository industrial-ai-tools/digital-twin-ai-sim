import numpy as np


def gaussian_random_field(size=64, alpha=2.0, a_min=0.1, a_max=3.0, seed=None):
    """Generate a smooth random permeability field."""
    if seed is not None:
        np.random.seed(seed)

    k = np.fft.fftfreq(size) * size
    kx, ky = np.meshgrid(k, k)
    k_norm = np.sqrt(kx**2 + ky**2)
    k_norm[0, 0] = 1.0

    power = (k_norm**2 + 1.0) ** (-alpha / 2.0)
    noise = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    field = np.fft.ifft2(np.fft.fft2(noise) * power).real

    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    field = field * (a_max - a_min) + a_min
    return field


def solve_darcy_2d(a_field, f=None, dx=1/64):
    """Solve -div(a * grad(u)) = f on [0,1]^2 with Dirichlet BC u=0."""
    from scipy.sparse import coo_matrix, csc_matrix
    from scipy.sparse.linalg import spsolve

    n = a_field.shape[0]
    if f is None:
        f = np.ones((n, n))
    h = dx
    N = (n - 2) * (n - 2)

    idx = np.arange(1, n - 1)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    row_idx = (I - 1) * (n - 2) + (J - 1)

    a_e = (a_field[1:-1, 1:-1] + a_field[1:-1, 2:]) / 2.0
    a_w = (a_field[1:-1, 1:-1] + a_field[1:-1, :-2]) / 2.0
    a_n = (a_field[1:-1, 1:-1] + a_field[:-2, 1:-1]) / 2.0
    a_s = (a_field[1:-1, 1:-1] + a_field[2:, 1:-1]) / 2.0

    center = a_e + a_w + a_n + a_s
    main_diag = center.reshape(-1) / (h * h)

    east_vals = -a_e[:, :-1].reshape(-1) / (h * h)
    east_cols = row_idx[:, :-1].reshape(-1)
    east_rows = row_idx[:, 1:].reshape(-1)

    west_vals = -a_w[:, 1:].reshape(-1) / (h * h)
    west_cols = row_idx[:, 1:].reshape(-1)
    west_rows = row_idx[:, :-1].reshape(-1)

    north_vals = -a_n[:-1, :].reshape(-1) / (h * h)
    north_cols = row_idx[1:, :].reshape(-1)
    north_rows = row_idx[:-1, :].reshape(-1)

    south_vals = -a_s[1:, :].reshape(-1) / (h * h)
    south_cols = row_idx[:-1, :].reshape(-1)
    south_rows = row_idx[1:, :].reshape(-1)

    A = coo_matrix((main_diag, (row_idx.reshape(-1), row_idx.reshape(-1))), shape=(N, N))
    A += coo_matrix((east_vals, (east_rows, east_cols)), shape=(N, N))
    A += coo_matrix((west_vals, (west_rows, west_cols)), shape=(N, N))
    A += coo_matrix((north_vals, (north_rows, north_cols)), shape=(N, N))
    A += coo_matrix((south_vals, (south_rows, south_cols)), shape=(N, N))

    A = csc_matrix(A)
    b = f[1:-1, 1:-1].reshape(-1)
    u_interior = spsolve(A, b)
    u = np.zeros((n, n))
    u[1:-1, 1:-1] = u_interior.reshape(n - 2, n - 2)
    return u


def generate_dataset(n_samples=200, res=64, save_dir="data"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    coeffs = np.zeros((n_samples, res, res))
    sols = np.zeros((n_samples, res, res))
    for i in range(n_samples):
        a_field = gaussian_random_field(size=res, seed=i)
        u = solve_darcy_2d(a_field)
        coeffs[i] = a_field
        sols[i] = u
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{n_samples}")
    np.save(f"{save_dir}/coeffs.npy", coeffs)
    np.save(f"{save_dir}/sols.npy", sols)
    print("Done.")