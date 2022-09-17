import numpy as np
import cvxpy as cp
import time

from validity import lower_validity_bound, \
    upper_validity_bound, zero_lower_validity_bound


def _heuristic_correction(x, mean, cov, epsilon):
    """_heuristic_correction.

    Parameters
    ----------
    x :
        x
    mean :
        mean
    cov :
        cov
    epsilon : float > 0
        epsilon
    """
    d = mean.shape[0]
    z = cp.Variable(d)
    t = cp.Variable(nonneg=True)

    constraints = [
        cp.SOC(epsilon * t, z - t * x),
        #         cp.norm(z - t * x) <= epsilon * t,
        cp.transpose(z) @ mean.reshape(-1, 1) == np.ones((1, 1)),
        z[0] == t
    ]

    objective = cp.Minimize(cp.quad_form(z, cov))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    return z.value / t.value


def mahalanobis_correction(cfs, mu_hat, Sigma_hat, rho, epsilon, max_k=None, logs={}):
    """mahalanobis_correction.
        Implementation of mahalanobis correction.

    Parameters
    ----------
    cfs : np.array of size (num_cfs, d)
        cfs is current counterfactual plan
    mu_hat : np.array of size (d)
        mu_hat: estimated mean of classifier's weights
    Sigma_hat : 
        Sigma_hat is estimated covariance matrix  of classifier's weights
    rho : a float > 0
        rho is Gelbrich bound
    epsilon : a float > 0
        epsilon is Gelbrich bound
    max_k : int > 0
        max_k is the number of used corrections
    logs :
        logs
    """
    num_cfs, d = cfs.shape
    if max_k is None:
        max_k = num_cfs

    new_cfs = np.copy(cfs)

    old_lb, opt_vars = lower_validity_bound(new_cfs, mu_hat, Sigma_hat, rho)
    lambd = opt_vars['lambda']
    chosen_ones = np.argsort(-lambd)[:max_k]

    for k in chosen_ones:
        x_k = new_cfs[k]
        # start = time.time()
        x_prime_k = _heuristic_correction(x_k, mu_hat, Sigma_hat, epsilon)
        # print("correction time: ", time.time() - start)
        new_cfs[k] = x_prime_k

    # start = time.time()
    lb, opt_vars = lower_validity_bound(new_cfs, mu_hat, Sigma_hat, rho)
    # print("compute L* time: ", time.time() - start)
    lambd = opt_vars['lambda']

    return new_cfs
