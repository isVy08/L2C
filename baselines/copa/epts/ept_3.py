import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
import random
import joblib
import dice_ml
from tqdm import tqdm
from functools import partialmethod

from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

from utils import helpers
from correction import mahalanobis_correction
from classifiers import LRSklearn, LRTorch
from validity import lower_validity_bound, upper_validity_bound
from utils import helpers
from utils.utils import check_symmetry, compute_diversity, compute_dpp, compute_proximity, pdump, pload
from utils.validation import check_random_state
from utils.data_transformer import DataTransformer
from epts.utils import get_pretrain
from epts.dice_wrapper import DicePyTorchWrapper
from epts.common import classifier_classes, method_funcs
from utils.utils import sqrtm_psd, gelbrich_dist

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def simulate_empirical_validity(cfs, mu, Sigma, sample_size, random_state=None):
    theta = multivariate_normal.rvs(
        mean=mu, cov=Sigma, size=sample_size, random_state=random_state)
    return np.sum(np.all(theta @ cfs.T >= 0, axis=1)) / sample_size


def eval_cfs_func(cfs, args):
    ec = args['ec']
    mu_hat = args['mu_hat']
    Sigma_hat = args['Sigma_hat']

    ret = {}
    # Mean shift ept
    rho = []
    validity = []
    alpha_list = np.linspace(ec.min_alpha, ec.max_alpha, ec.num_alpha)
    mu_bar = np.array([0, -1, 0])
    p1_sample_size = 1000000

    for alpha in alpha_list:
        mu_1 = mu_hat + alpha * mu_bar
        Sigma_1 = Sigma_hat
        val = simulate_empirical_validity(cfs, mu_1, Sigma_1, p1_sample_size)
        d = gelbrich_dist(mu_hat, Sigma_hat, mu_1, Sigma_1)
        rho.append(d)
        validity.append(val)

    ret['mean_shift'] = {'rho': rho, 'validity': validity}

    # Cov shift
    rho = []
    validity = []
    beta_list = np.linspace(ec.min_beta, ec.max_beta, ec.num_beta)

    for beta in beta_list:
        mu_1 = mu_hat
        Sigma_1 = (1 + beta) * np.identity(mu_hat.shape[0])
        val = simulate_empirical_validity(cfs, mu_1, Sigma_1, p1_sample_size)
        d = gelbrich_dist(mu_hat, Sigma_hat, mu_1, Sigma_1)
        rho.append(d)
        validity.append(val)

    ret['cov_shift'] = {'rho': rho, 'validity': validity}

    # mean & cov shift
    rho = []
    validity = []

    for alpha, beta in zip(alpha_list, beta_list):
        mu_1 = mu_hat + alpha * mu_bar
        Sigma_1 = (1 + beta) * np.identity(mu_hat.shape[0])
        val = simulate_empirical_validity(cfs, mu_1, Sigma_1, p1_sample_size)
        d = gelbrich_dist(mu_hat, Sigma_hat, mu_1, Sigma_1)
        rho.append(d)
        validity.append(val)

    ret['mean_cov_shift'] = {'rho': rho, 'validity': validity}

    return ret


def __run_method_on_dataset(dname, ec, wdir, mname, cname,
                            num_proc=1, seed=None, logger=None):
    torch.manual_seed(seed + 7)
    np.random.seed(seed + 8)
    random.seed(seed + 9)
    logger.info("Running classifier %s; method %s; dataset %s...",
                cname, mname, dname)
    # setting parameters; only used in synthetic data
    beta = 1
    sp = ec.ept3.synthetic_params
    mean_0 = np.array(sp['mean_0'])
    mean_1 = np.array(sp['mean_1'])
    cov_0 = beta * np.array(sp['cov_0'])
    cov_1 = np.array(sp['cov_1'])
    params = (sp['sample_size'], mean_0, cov_0,
              mean_1, cov_1, sp['seed'])

    # getting dataset
    dataset, numerical = helpers.get_dataset(dname, params=params)
    full_dataset, full_numerical = helpers.get_full_dataset(
        dname, params=params)

    full_dice_data = dice_ml.Data(dataframe=full_dataset,
                                  continuous_features=full_numerical,
                                  outcome_name='label')

    y = dataset['label']
    X = dataset.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1, stratify=y)
    X_test = X_test.append(X_train)
    y_test = y_test.append(y_train)

    classifier = classifier_classes[cname]
    transformer = DataTransformer(full_dice_data)
    clf = classifier(transformer)
    clf.train(X_train, y_train, max_iter=1000)

    y_pred = clf.predict(X_test)
    theta0 = clf.weights

    d = theta0.shape[0]
    mu_hat = theta0
    Sigma_hat = ec.ept3.Sigma_hat_beta * np.identity(d)

    dice_model = dice_ml.Model(
        model=clf.get_model(), backend=classifier.backend)
    if isinstance(clf, LRTorch):
        dice = DicePyTorchWrapper(full_dice_data, dice_model)
    else:
        dice = dice_ml.Dice(
            full_dice_data, dice_model, method=ec.ept3.dice_method)

    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
    uds_X, uds_y = uds_X.head(
        ec.ept3.max_test_ins), uds_y.head(ec.ept3.max_test_ins)
    num_cfs = ec.ept3.num_cfs

    method = method_funcs[mname]

    params = {}
    params['dname'] = dname
    params['cname'] = cname
    params['mname'] = mname

    job_args = []

    eval_args = {
        "mu_hat": mu_hat,
        "Sigma_hat": Sigma_hat,
        "ec": ec.ept3,
    }
    for idx in range(len(uds_X)):
        test_ins_df = uds_X[idx:idx+1]

        job_args.append((idx, test_ins_df, num_cfs, ec.ept3, dice, mu_hat, Sigma_hat,
                         transformer, None, seed, logger, params, eval_cfs_func, eval_args))

    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(method)(
        *args) for args in job_args)

    combined_rets = {'mean_shift': {}, 'cov_shift': {}, 'mean_cov_shift': {}}

    m_rho, m_val = [], []
    c_rho, c_val = [], []
    mc_rho, mc_val = [], []

    for ret in rets:
        m_rho.append(ret['mean_shift']['rho'])
        m_val.append(ret['mean_shift']['validity'])
        c_rho.append(ret['cov_shift']['rho'])
        c_val.append(ret['cov_shift']['validity'])
        mc_rho.append(ret['mean_cov_shift']['rho'])
        mc_val.append(ret['mean_cov_shift']['validity'])

    combined_rets = {
        'mean_shift': {
            'rho': np.array(m_rho),
            'validity': np.array(m_val)
        },
        'cov_shift': {
            'rho': np.array(c_rho),
            'validity': np.array(c_val)
        },
        'mean_cov_shift': {
            'rho': np.array(mc_rho),
            'validity': np.array(mc_val)
        },
    }

    pdump(combined_rets, f'{dname}_{mname}_{cname}.pickle', wdir)
    logger.info("==> Done classifier %s; method %s; dataset: %s",
                cname, mname, dname)


def __plot_1(dname, cname, methods, wdir, ec):
    rets = {}
    for mname in methods:
        rets[mname] = pload(f'{dname}_{mname}_{cname}.pickle', wdir)

    plt.style.use('seaborn-white')
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    shift_types = ['mean_shift', 'cov_shift', 'mean_cov_shift']
    shift_map = {'mean_shift': 'Mean shift',
                 'cov_shift': 'Covariance shift',
                 'mean_cov_shift': 'Mean & Covariance shift'}
    method_map = {'dice': 'DiCE',
                  'mahalanobis': 'MahalanobisCrr', 'pgd': 'COPA'}

    pad_x = 8e-3 * np.array([-1, 0, 1])
    joint_mname = '_'.join(methods)
    plt.rcParams.update({'font.size': 17})
    plt.rcParams.update({'axes.linewidth': 2})
    plt.rcParams.update({'lines.linewidth': 2})

    for i in range(len(shift_types)):
        fig, ax = plt.subplots(figsize=(6, 4))
        for j, mname in enumerate(methods):
            x = np.mean(rets[mname][shift_types[i]]['rho'], axis=0)
            y_mean = np.mean(rets[mname][shift_types[i]]['validity'], axis=0)
            y_std = np.std(rets[mname][shift_types[i]]['validity'], axis=0)
            print(rets[mname][shift_types[i]]['validity'].shape)
            # print(x, y_mean)
            ax.errorbar(x + pad_x[j] * np.max(x) * np.ones_like(x), y_mean, yerr=y_std, label=method_map[mname])

        ax.legend(loc='lower left')
        ax.set_xlabel('$\mathbb{G}(\hat{\mu}, \hat{\Sigma}, \mu_g, \Sigma_g)$')
        # ax.set_title(shift_map[shift_types[i]])
        ax.grid(b=True)
        ax.set_ylabel('Validity')
        filename = os.path.join(wdir, f'{dname}_{joint_mname}_{shift_types[i]}.png')
        plt.savefig(filename, dpi=800, bbox_inches='tight')


    plt.tight_layout(pad=5.0)
    joint_mname = '_'.join(methods)
    filename = os.path.join(wdir, f'{dname}_{joint_mname}.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight')


def run_ept_3(ec, wdir, datasets, methods, classifiers,
              num_proc=4, plot_only=False, seed=None, logger=None):
    logger.info("Running ept 3")

    if datasets is None or len(datasets) == 0:
        datasets = ec.ept3.all_datasets

    if methods is None or len(methods) == 0:
        methods = ec.ept3.all_methods

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.ept3.all_classifiers

    for cname in classifiers:
        for dname in datasets:
            for mname in methods:
                if not plot_only:
                    __run_method_on_dataset(
                        dname, ec, wdir, mname,
                        cname, num_proc, seed, logger)

            __plot_1(dname, cname, methods, wdir, ec)

    # __summarize(datasets, methods, classifiers, wdir, ec)
