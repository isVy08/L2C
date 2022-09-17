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


def __run_method_on_dataset(dname, ec, wdir, mname, cname,
                            num_proc=1, seed=None, logger=None):
    torch.manual_seed(seed + 7)
    np.random.seed(seed + 8)
    random.seed(seed + 9)
    logger.info("Running classifier %s; method %s; dataset %s...",
                cname, mname, dname)
    # setting parameters; only used in synthetic data
    sp = ec.ept2.synthetic_params
    params = (sp['sample_size'], sp['mean_0'], sp['cov_0'],
              sp['mean_1'], sp['cov_1'], sp['seed'])

    # getting dataset
    dataset, numerical = helpers.get_dataset(dname, params=params)
    full_dataset, full_numerical = helpers.get_full_dataset(dname, params=params)

    # use for transformer
    full_dice_data = dice_ml.Data(dataframe=full_dataset,
                             continuous_features=full_numerical,
                             outcome_name='label')

    y = dataset['label']
    X = dataset.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1, stratify=y)
    X_test = X_test.append(X_train)
    y_test = y_test.append(y_train)

    shifted_clfs = get_pretrain(cname, dname + '_shift', ec.ept2.num_clfs)
    org_clfs = get_pretrain(cname, dname, ec.ept2.num_clfs)

    org_theta = np.stack([org_clfs[i].weights for i in range(len(org_clfs))], axis = 1)
    shifted_theta = np.stack([shifted_clfs[i].weights for i in range(len(shifted_clfs))], axis = 1)


    mu_1 = np.mean(shifted_theta, axis=1)
    Sigma_1 = np.cov(shifted_theta)

    classifier = classifier_classes[cname]
    transformer = DataTransformer(full_dice_data)
    clf = classifier(transformer)
    clf.train(X_train, y_train, max_iter=2000)

    mu_hat = np.mean(org_theta, axis=1)
    Sigma_hat = 1.5 * np.cov(org_theta)

    y_pred = clf.predict(X_test)
    theta0 = clf.weights

    Sigma_hat_inv_sqrtm = np.linalg.inv(sqrtm_psd(Sigma_hat))
    Sigma_1_inv_sqrtm = np.linalg.inv(sqrtm_psd(Sigma_1))

    dice_model = dice_ml.Model(model=clf.get_model(), backend=classifier.backend)
    if isinstance(clf, LRTorch):
        dice = DicePyTorchWrapper(full_dice_data, dice_model)
    else:
        dice = dice_ml.Dice(
            full_dice_data, dice_model, method=ec.ept2.dice_method)

    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
    uds_X, uds_y = uds_X.head(
        ec.ept2.max_test_ins), uds_y.head(ec.ept2.max_test_ins)
    num_cfs = ec.ept2.num_cfs

    method = method_funcs[mname]

    params = {}
    params['dname'] = dname
    params['cname'] = cname
    params['mname'] = mname

    job_args = []

    for idx in range(len(uds_X)):
        test_ins_df = uds_X[idx:idx+1]

        job_args.append((idx, test_ins_df, num_cfs, ec.ept2, dice, mu_hat, Sigma_hat,
                         transformer, shifted_clfs, seed, logger, params))

    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(method)(
        *args) for args in job_args)

    dpp = []
    diversity = []
    prox = []
    lb = []
    validity = []
    current_validity = []

    for ret in rets:
        v, l, dvs, dp, prx, cv = ret
        dpp.append(dp)
        diversity.append(dvs)
        prox.append(prx)
        lb.append(l)
        validity.append(v)
        current_validity.append(cv)

    data = (np.array(validity), np.array(lb), np.array(diversity),
            np.array(dpp), np.array(prox), np.array(current_validity))
    print(data)
    pdump(data, f'{dname}_{mname}_{cname}.pickle', wdir)
    logger.info("==> Done classifier %s; method %s; dataset: %s",
                cname, mname, dname)


def __summarize(datasets, methods, classifiers, wdir, ec):
    def to_mean_std(m, s, is_best):
        if is_best:
            return "\\textbf{" + "{:.3f}".format(m) + "}" + " $\pm$ {:.3f}".format(s)
        else:
            return "{:.3f} $\pm$ {:.3f}".format(m, s)

    cname_map = {'logistic': 'lr', 'nn': 'nn', 'lrt': 'lrt'}
    pgd_methods = ['pgd'] + ['pgd' + str(i) for i in range(1, 10)]
    method_map = {
        'dice': 'DiCE',
        'mahalanobis': "MahalanobisCrr",
        'gradient': 'DroDiceGD',
    }
    for m in pgd_methods:
        lamb_1 = ec.ept2.params[m]['robust_weight']
        lamb_2 = ec.ept2.params[m]['diversity_weight']
        method_map[m] = f"COPA ($\lambda_1 = {lamb_1}; \lambda_2 = {lamb_2}$)"
    dataset_map = {
        "synthesis": "Synthetic",
        "german": "Correction",
        "sba": "Temporal",
        "student": "Geospatial"
    }
    metrics = ['prox', 'dpp', 'lvb', 'tv', 'cv']
    metric_order = {'prox': -1, 'dvs': 1, 'lvb': 1, 'tv': 1, 'dpp': 1, 'cv': 1}
    data_df = {"ds": [], "mt": []}

    for cname in classifiers:
        for metric in metrics:
            data_df[f"{cname_map[cname]}{metric}"] = []

    data = {}
    for dname in datasets:
        data[dname] = {}
        for cname in classifiers:
            data[dname][cname] = {}
            data[dname][cname]['best'] = {}
            for metric in metrics:
                data[dname][cname]['best'][metric] = -np.inf

            for i, mname in enumerate(methods):
                name = f'{dname}_{mname}_{cname}.pickle'
                tv, lvb, dvs, dpp, prox, cv = pload(name, wdir)
                print(dname, mname, cname, "tv:", tv, "lvb:", lvb)
                r = {'tv': tv, 'dvs': dvs, 'dpp': dpp, 'prox': prox, 'lvb': lvb, 'cv': cv}
                data[dname][cname][mname] = {}
                for metric in metrics:
                    m = np.mean(r[metric])
                    s = np.std(r[metric])
                    data[dname][cname][mname][metric] = (m, s)
                    data[dname][cname]['best'][metric] = max(data[dname][cname]['best'][metric],
                                                                m * metric_order[metric])

            for i, mname in enumerate(methods):
                data_df['ds'].append(dataset_map[dname] if i == 0 else '')
                data_df['mt'].append(method_map[mname])
                for metric in metrics:
                    m, s = data[dname][cname][mname][metric]
                    is_best = (m * metric_order[metric] == data[dname][cname]['best'][metric])
                    data_df[f"{cname_map[cname]}{metric}"].append(to_mean_std(m, s, is_best))

    df = pd.DataFrame(data_df)
    df.to_csv(os.path.join(wdir, 'realdata.csv'), index=False)


def run_ept_2(ec, wdir, datasets, methods, classifiers,
              num_proc=4, plot_only=False, seed=None, logger=None):
    logger.info("Running ept 2")

    if datasets is None or len(datasets) == 0:
        datasets = ec.ept2.all_datasets

    if methods is None or len(methods) == 0:
        methods = ec.ept2.all_methods

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.ept2.all_classifiers

    for cname in classifiers:
        for dname in datasets:
            for mname in methods:
                if not plot_only:
                    __run_method_on_dataset(
                        dname, ec, wdir, mname,
                        cname, num_proc, seed, logger)

    __summarize(datasets, methods, classifiers, wdir, ec)
