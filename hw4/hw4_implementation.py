import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.sparse


def compute_Q(n):
    # Difference matrix Q for 1D TV regularization
    return scipy.sparse.diags([1, -1], [0, 1], shape=(n - 1, n), format='csc')


def admm_solve(f, lmbda, rho, n_iter=100):
    n = len(f)
    Q = compute_Q(n)
    x = np.zeros(n)
    z = np.zeros(n - 1)
    y = np.zeros(n - 1)

    for _ in range(n_iter):
        # x-update adjusted for the factor 1/(2*lambda) in J_f
        A = scipy.sparse.eye(n) + lmbda * rho * Q.T @ Q
        b = f + lmbda * Q.T @ (rho * z - y)
        x = scipy.sparse.linalg.spsolve(A, b)

        # z-update: soft thresholding
        Qx = Q @ x
        z = np.maximum(0, np.abs(Qx + y / rho) - 1 / rho) * np.sign(Qx + y / rho)

        # y-update
        y += rho * (Qx - z)

    return x


def run_experiment(n, lmbda, rho):
    x = np.linspace(0, 1, n)
    # f = 125 + 100 * np.sin(5 * x)
    f = 125 + 100 * scipy.signal.square(np.sin(5*x))

    # indices, starting from 1 to avoid division by zero
    indices = np.arange(1, n + 1)

    ## Note in the noise generation, scalings were adjusted manually as n increased

    # Strong convergence noise
    # noise_strong = 50 * np.sin(2 * np.pi * indices * x) / indices
    noise_strong = 150 * np.random.normal(0, 1, n) / n
    fn_strong = f + noise_strong

    # Weak convergence noise
    noise_weak = 10 * np.sin(2 * np.pi * indices * x)
    fn_weak = f + noise_weak

    # Solve using ADMM
    u_clean = admm_solve(f, lmbda, rho)
    u_strong = admm_solve(fn_strong, lmbda, rho)
    u_weak = admm_solve(fn_weak, lmbda, rho)

    # Plot results
    plt.figure(figsize=(18, 6))
    plt.suptitle(f'Results for n = {n}', fontsize=16)

    plt.subplot(131)
    plt.plot(x, f, label='Original f')
    plt.plot(x, u_clean, label='Clean Solution')
    plt.legend()
    plt.title('Clean Data')

    plt.subplot(132)
    plt.plot(x, fn_strong, label='Noisy f (Strong)')
    plt.plot(x, u_strong, label='Solution to Strong Noise')
    plt.legend()
    plt.title("Strongly Convergent Noise")

    plt.subplot(133)
    plt.plot(x, fn_weak, label='Noisy f (Weak)')
    plt.plot(x, u_weak, label='Solution to Weak Noise')
    plt.legend()
    plt.title("Weakly Convergent Noise")

    plt.show()

# Can adjust parameters n, lambda, and rho
run_experiment(50, 10, 1)
