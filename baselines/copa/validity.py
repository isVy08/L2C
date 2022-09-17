import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm


def lower_validity_bound(cfs, est_mean, est_cov, rho, verbose=False):
    """lower_bound.
    """
    num_cfs, d = cfs.shape
    # define variables and parameters
    mu = cp.Variable((d, 1))  # in R^d
    Sigma = cp.Variable((d, d), PSD=True)  # \in S^d_+
    M = cp.Variable((d, d), PSD=True)  # \in S^d_+
    C = cp.Variable((d, d))
    lambd, z, Z = [], [], []
    for j in range(num_cfs):
        lambd.append(cp.Variable())
        z.append(cp.Variable((d, 1)))
        Z.append(cp.Variable((d, d), symmetric=True))  # \in S^d

    # constraints
    constraints = []

    # -x_j^t * z_j >= 0  \forall j \in [J]
    constraints += [
        (-cp.sum(cp.multiply(z[j], cfs[j].reshape((-1, 1)))) >= 0) for j in range(num_cfs)
    ]

    # [[Z_j z_j] [z_j^T lambda_j] >> 0 \forall j \in [J]
    constraints += [
        (cp.bmat([[Z[j], z[j]], [cp.transpose(z[j]), cp.reshape(lambd[j], (1, 1))]]) >> 0) for j in range(num_cfs)
    ]

    # \sum_{j\in[J]} [[Z_j z_j] [z_j^T \lambda_j] << [[M mu] [mu_T 1]]
    constraints += [
        (sum([
            cp.bmat([[Z[j], z[j]], [cp.transpose(z[j]), cp.reshape(lambd[j], (1, 1))]]) for j in range(num_cfs)
        ])
            << cp.bmat([[M, mu], [cp.transpose(mu), cp.reshape(1, (1, 1))]]))
    ]

    # \norm(\hat{mu})^2 - 2 \hat{mu}^T mu + trace[M + \hat{Sigma} - 2C] <= rho^2
    constraints += [
        cp.square(cp.norm(est_mean)) - 2 * cp.sum(cp.multiply(
            est_mean.reshape(-1, 1), mu)) + cp.trace(M + est_cov - 2*C)
        <= rho * rho
    ]

    # [[Sigma C] [C^T \hat{Sigma}] >> 0
    constraints += [
        cp.bmat([[Sigma, C], [cp.transpose(C), est_cov]]) >> 0
    ]

    # [[M-Sigma mu] [mu^T 1] >> 0
    constraints += [
        cp.bmat(
            [[M - Sigma, mu], [cp.transpose(mu), cp.reshape(1, (1, 1))]]) >> 0
    ]

    obj = cp.Minimize(1 - sum(lambd))

    lb_prob = cp.Problem(obj, constraints)
    lb_prob.solve(solver=cp.MOSEK, verbose=False)

    if verbose:
        print("==> Testing with:")
        print("Counterfactual plan:\n", cfs)
        print("theta_mean (mu^\\hat):\n", est_mean)
        print("theta_cov (Sigma^\\hat:\n", est_cov)
        print("rho: ", rho)
        print("="*10)
        print("Prob's status:", lb_prob.status)
        print("lower bound: ", lb_prob.value)
        print("mu:", mu.value)
        print("Sigma:", Sigma.value)
        print("lambda:", *[e.value for e in lambd])
        print("z:", *[e.value for e in z])
        print("Z:", *[e.value for e in Z])
        print("M:", M.value)
        print("C:", C.value)
        # print("Gelbrich distance:", gelbrich_dist(est_mean, est_cov, mu.value.squeeze(), Sigma.value))
        # print("Inequality 3 ( ... <= rho^2):", (np.linalg.norm(est_mean) ** 2 - 2 * np.inner(est_mean.reshape(-1), mu.value.reshape(-1)) + np.trace(M.value + est_cov - 2 * C.value) - rho ** 2))
        # print("M - mu*mu' - Sigma:\n", M.value - Sigma.value - mu.value @ mu.value.transpose())

    optvars = {}
    optvars['mu'] = mu.value
    optvars['Sigma'] = Sigma.value
    optvars['lambda'] = np.array([e.value for e in lambd])
    optvars['z'] = [e.value for e in z]
    optvars['Z'] = [e.value for e in Z]
    optvars['M'] = M.value
    optvars['C'] = C.value

    return lb_prob.value, optvars


def upper_validity_bound(cfs, est_mean, est_cov, rho, verbose=False):
    """upper_bound.

    """
    num_cfs, d = cfs.shape
    sqrtm_est_cov = sqrtm(est_cov)
    # assert check_symmetry(sqrtm_est_cov)

    # defining variables
    gamma = cp.Variable(nonneg=True)
    z0 = cp.Variable()
    z = cp.Variable((d, 1))
    Z = cp.Variable((d, d), PSD=True)
    q = cp.Variable(nonneg=True)
    Q = cp.Variable((d, d), PSD=True)
    lambd = [cp.Variable(nonneg=True) for i in range(num_cfs)]

    # defining constraints
    constraints = []

    at = gamma * np.identity(d) - Z
    bt = gamma * sqrtm_est_cov
    ct = gamma * sqrtm_est_cov
    dt = Q
    constraints += [
        cp.bmat([[at, bt], [ct, dt]]) >> 0
    ]

    at = gamma * np.identity(d) - Z
    bt = gamma * est_mean.reshape(-1, 1) + z
    ct = gamma * est_mean.reshape(1, -1) + cp.transpose(z)
    dt = cp.reshape(q, (1, 1))
    constraints += [
        cp.bmat([[at, bt], [ct, dt]]) >> 0
    ]

    constraints += [
        cp.bmat([[Z, z], [cp.transpose(z), cp.reshape(z0, (1, 1))]]) >> 0
    ]

    lhs = cp.bmat([[Z, z], [cp.transpose(z), cp.reshape(z0 - 1, (1, 1))]])
    rhs = sum([
        lambd[j] * cp.bmat([[np.zeros((d, d)), 0.5 * cfs[j].reshape(-1, 1)],
                            [0.5 * cfs[j].reshape(1, -1), np.zeros((1, 1))]])
        for j in range(num_cfs)
    ])
    constraints += [lhs >> rhs]

    objective = cp.Minimize(
        z0 + gamma * (rho ** 2 - np.linalg.norm(est_mean) ** 2 - np.trace(est_cov)) +
        q + cp.trace(Q)
    )

    ub_prob = cp.Problem(objective, constraints)

    ub_prob.solve(solver=cp.MOSEK, verbose=False)

    if verbose:
        print("==> Testing with:")
        print("Counterfactual plan:\n", cfs)
        print("theta_mean (mu^\\hat):\n", est_mean)
        print("theta_cov (Sigma^\\hat:\n", est_cov)
        print("rho: ", rho)
        print("="*10)
        print("Status: ", ub_prob.status)
        print("Upper bound:", ub_prob.value)
        print("gamma: ", gamma.value)
        print("z0: ", z0.value)
        print("z: \n", z.value)
        print("Z: \n", Z.value)
        print("q: ", q.value)
        print("Q: \n", Q.value)
        print("lambda: \n", *[e.value for e in lambd])

    optvars = {}
    optvars['gamma'] = gamma.value
    optvars['z0'] = z0.value
    optvars['z'] = z.value
    optvars['Z'] = Z.value
    optvars['q'] = q.value
    optvars['Q'] = Q.value
    optvars['lambda'] = np.array([e.value for e in lambd])

    return ub_prob.value, optvars


def zero_lower_validity_bound(cfs, est_mean, est_cov, rho=0, verbose=False):
    """lower_bound.
    """
    num_cfs, d = cfs.shape
    # define variables and parameters
    lambd, z, Z = [], [], []
    for j in range(num_cfs):
        lambd.append(cp.Variable())
        z.append(cp.Variable((d, 1)))
        Z.append(cp.Variable((d, d), symmetric=True))  # \in S^d

    # constraints
    constraints = []

    # -x_j^t * z_j >= 0  \forall j \in [J]
    constraints += [
        (-cp.sum(cp.multiply(z[j], cfs[j].reshape((-1, 1)))) >= 0) for j in range(num_cfs)
    ]

    # [[Z_j z_j] [z_j^T lambda_j] >> 0 \forall j \in [J]
    constraints += [
        (cp.bmat([[Z[j], z[j]], [cp.transpose(z[j]), cp.reshape(lambd[j], (1, 1))]]) >> 0) for j in range(num_cfs)
    ]

    # \sum_{j\in[J]} [[Z_j z_j] [z_j^T \lambda_j] << [[M mu] [mu_T 1]]
    constraints += [
        (sum([
            cp.bmat([[Z[j], z[j]], [cp.transpose(z[j]), cp.reshape(lambd[j], (1, 1))]]) for j in range(num_cfs)
        ])
            << cp.bmat([[est_cov + est_mean.reshape(-1, 1) @ est_mean.reshape(1, -1), est_mean.reshape(-1, 1)],
                        [est_mean.reshape(1, -1), cp.reshape(1, (1, 1))]]))
    ]

    obj = cp.Minimize(1 - sum(lambd))

    lb_prob = cp.Problem(obj, constraints)
    lb_prob.solve(solver=cp.MOSEK, verbose=False)

    if verbose:
        print("==> Testing with:")
        print("Counterfactual plan:\n", cfs)
        print("theta_mean (mu^\\hat):\n", est_mean)
        print("theta_cov (Sigma^\\hat:\n", est_cov)
        print("rho: ", rho)
        print("="*10)
        print("Prob's status:", lb_prob.status)
        print("lower bound: ", lb_prob.value)
        print("lambda:", *[e.value for e in lambd])
        print("z:", *[e.value for e in z])
        print("Z:", *[e.value for e in Z])
        # print("Gelbrich distance:", gelbrich_dist(est_mean, est_cov, mu.value.squeeze(), Sigma.value))
        # print("Inequality 3 ( ... <= rho^2):", (np.linalg.norm(est_mean) ** 2 - 2 * np.inner(est_mean.reshape(-1), mu.value.reshape(-1)) + np.trace(M.value + est_cov - 2 * C.value) - rho ** 2))
        # print("M - mu*mu' - Sigma:\n", M.value - Sigma.value - mu.value @ mu.value.transpose())

    optvars = {}
    optvars['lambda'] = np.array([e.value for e in lambd])
    optvars['z'] = [e.value for e in z]
    optvars['Z'] = [e.value for e in Z]

    return lb_prob.value, optvars
