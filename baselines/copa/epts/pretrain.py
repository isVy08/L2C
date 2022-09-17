import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import joblib
import copy
import dice_ml
from tqdm import tqdm
from functools import partialmethod

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, \
    classification_report, confusion_matrix, roc_auc_score

from utils import helpers
from correction import mahalanobis_correction
from classifiers import LRSklearn, LRTorch
from validity import lower_validity_bound, upper_validity_bound
from utils import helpers
from utils.utils import check_symmetry, compute_diversity, compute_dpp, compute_proximity, pdump, pload
from utils.validation import check_random_state
from utils.data_transformer import DataTransformer
from epts.common import classifier_classes


def performance(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= 0.5).astype(np.float64)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    # print(accracy, auc)
    return accuracy, auc


def train_clf(i, X, y, full_dice_data, classifier, logger):
    logger.info("Training classifier %d", i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=i, stratify=y)

    # train a classifier
    transformer = DataTransformer(full_dice_data)
    clf = classifier(transformer)
    clf.train(X_train, y_train, num_epoch=1500, verbose=False)
    acc, auc = performance(clf, X_test, y_test)

    return clf, acc, auc


def pretrain_classifier_on_dataset(ec, wdir, dname, cname, num_proc, logger):
    logger.info("Training classifier %s on dataset %s", cname, dname)
    sp = ec.ept2.shifted_synthetic_params
    params = (sp['sample_size'], sp['mean_0'], sp['cov_0'],
              sp['mean_1'], sp['cov_1'], sp['seed'])

    # getting dataset
    dataset, numerical = helpers.get_dataset(dname, params=params)

    # if dname == 'sba_shift':
        # org_dataset, _ = helpers.get_dataset(dname, params=params)
        # dataset = dataset.append(org_dataset)

    dice_data = dice_ml.Data(dataframe=dataset,
                             continuous_features=numerical,
                             outcome_name='label')

    full_dataset, full_numerical = helpers.get_full_dataset(dname)
    full_dice_data = dice_ml.Data(dataframe=full_dataset,
                             continuous_features=full_numerical,
                             outcome_name='label')

    y = dataset['label']
    X = dataset.drop('label', axis=1)

    classifier = classifier_classes[cname]

    clfs = []
    auc_list = []
    acc_list = []

    job_args = []

    for i in range(ec.pretrain.num_classifiers):
        job_args.append((i, X, y, full_dice_data, classifier, logger))

    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(train_clf)(
        *args) for args in job_args)

    for ret in rets:
        clf, acc, auc = ret
        auc_list.append(auc)
        acc_list.append(acc)
        clfs.append(copy.deepcopy(clf))

    name = f"{cname}_{dname}_{ec.pretrain.num_classifiers}.pickle"
    pdump(clfs, name, wdir)

    log_file = os.path.join(wdir, f"{cname}_{dname}_{ec.pretrain.num_classifiers}.txt")
    with open(log_file, mode='w') as f:
        f.write("auc mean: {}\n".format(np.mean(auc_list)))
        f.write("auc std: {}\n".format(np.std(auc_list)))
        f.write("acc mean: {}\n".format(np.mean(acc_list)))
        f.write("acc std: {}\n".format(np.std(acc_list)))


def pretrain_classifiers(ec, wdir, datasets, classifiers, num_proc=1, logger=None):
    logger.info("Training classifier...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.pretrain.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.pretrain.all_classifiers

    for classifier_name in classifiers:
        for dataset_name in datasets:
            pretrain_classifier_on_dataset(ec, wdir, dataset_name,
                                           classifier_name, num_proc, logger)
    pass
