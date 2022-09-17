import numpy as np
import math
import pickle
import os
from scipy.linalg import sqrtm, eigh

def gelbrich_dist(mean_0, cov_0, mean_1, cov_1):
    t1 = np.linalg.norm(mean_0 - mean_1)
    t2 = np.trace(cov_0 + cov_1 - 2 *
                  sqrtm(sqrtm(cov_1) @ cov_0 @ sqrtm(cov_1)))
    return math.sqrt(t1 ** 2 + t2)


def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def sqrtm_psd(A, check_finite=True):
    A = np.asarray(A)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    w, v = eigh(A, check_finite=check_finite)
    w = np.maximum(w, 0)
    return (v * np.sqrt(w)).dot(v.conj().T)


def lp_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=p)


def compute_dist(x, y, feature_weights=None):
    if feature_weights is None:
        feature_weights = np.ones_like(x)
    return np.sum(np.abs(x-y) * feature_weights, axis=0)


def compute_dpp(cfs, method='inverse_dist', dist=lp_dist):
    """Computes the DPP of a matrix."""
    num_cfs, d = cfs.shape
    det_entries = np.ones((num_cfs, num_cfs))
    if method == "inverse_dist":
        for i in range(num_cfs):
            for j in range(num_cfs):
                det_entries[(i, j)] = 1.0 / \
                    (1.0 + dist(cfs[i], cfs[j]))
                if i == j:
                    det_entries[(i, j)] += 0.0001

    elif method == "exponential_dist":
        for i in range(num_cfs):
            for j in range(num_cfs):
                det_entries[(i, j)] = 1.0 / \
                    (np.exp(dist(cfs[i], cfs[j])))
                if i == j:
                    det_entries[(i, j)] += 0.0001

    diversity_loss = np.linalg.det(det_entries)
    return diversity_loss


def compute_diversity(cfs, dice_data, weights='inverse_mad', intercept_feature=True):
    num_cfs, d = cfs.shape

    if weights == 'inverse_mad':
        feature_weights_dict = {}
        normalized_mads = dice_data.get_valid_mads(normalized=True)
        for feature in normalized_mads:
            feature_weights_dict[feature] = round(
                1/normalized_mads[feature], 2)

        feature_weights = [1.0] if intercept_feature else []
        for feature in dice_data.ohe_encoded_feature_names:
            if feature in feature_weights_dict:
                feature_weights.append(feature_weights_dict[feature])
            else:
                feature_weights.append(1.0)
        feature_weights = np.array(feature_weights)

    elif isinstance(weights, np.ndarray):
        feature_weights = weights
    else:
        feature_weights = np.ones(d)

    ret = 0
    for i in range(num_cfs):
        for j in range(i+1, num_cfs):
            # ret += compute_dist(cfs[i], cfs[j], feature_weights)
            ret += lp_dist(cfs[i], cfs[j], 2)

    return ret / (num_cfs * (num_cfs-1) / 2)


def compute_proximity(test_ins, cfs):
    num_cfs, d = cfs.shape
    ret = 0
    for i in range(num_cfs):
        ret += lp_dist(cfs[i], test_ins, 2)
    return ret / num_cfs


def compute_validity(cfs, shifted_clfs):
    num_valid = 0
    for clf in shifted_clfs:
        out = clf.predict(cfs[:, 1:], transform_data=False)
        if np.all( out == 1 ):
            num_valid += 1

    return num_valid / len(shifted_clfs)


def pdump(x, name, outdir='.'):
    with open(os.path.join(outdir, name), mode='wb') as f:
        pickle.dump(x, f)


def pload(name, outdir='.'):
    with open(os.path.join(outdir, name), mode='rb') as f:
        return pickle.load(f)
