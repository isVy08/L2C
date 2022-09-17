import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import joblib
import dice_ml
import random
from tqdm import tqdm
from functools import partialmethod

from sklearn.model_selection import train_test_split

from epts.common import classifier_classes, method_funcs
from utils import helpers
from correction import mahalanobis_correction
from classifiers.lr_sklearn import LRSklearn
from validity import lower_validity_bound, upper_validity_bound
from utils import helpers
from utils.utils import check_symmetry, compute_diversity, compute_dpp, compute_proximity, pdump, pload
from utils.validation import check_random_state
from utils.data_transformer import DataTransformer
from classifiers import LRTorch
from epts.utils import get_pretrain
from epts.dice_wrapper import DicePyTorchWrapper
from epts.common import _project

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def __run_test_ins(idx, test_ins_df, method, model, cfgenerator,
                   transformer, mu_hat, Sigma_hat, ec, logger,
                   vary_k=True, vary_eps=True):
    # get parameters
    num_cfs = ec.num_cfs

    cfps = cfgenerator.generate_counterfactuals(test_ins_df, total_CFs=num_cfs,
                                                desired_class="opposite",
                                                posthoc_sparsity_param=None)

    cfp = cfps.cf_examples_list[0]

    test_ins = cfp.test_instance_df
    cf_plan = cfp.final_cfs_df
    cfs = cfp.final_cfs_df_sparse
    cfs = np.hstack([np.ones((num_cfs, 1)), cfs])
    # print(cfs)
    cfs = _project(cfs, mu_hat, ec.epsilon_pgd)

    # transforming categorical features if exist
    test_ins = transformer.transform(test_ins.drop('label', axis=1),
                                     intercept_feature=True, tonumpy=True).squeeze()

    ret_k, ret_eps = None, None
    # varying k
    if vary_k:
        logger.info("Run instance %d: varying k", idx)
        k_list = []
        diversity_list = []
        dpp_list = []
        lb_list = []
        prox_list = []
        d = cfs.shape[1]
        rho = ec.rho
        epsilon = ec.epsilon

        for k in range(0, num_cfs+1):
            new_cfs = mahalanobis_correction(
                cfs, mu_hat, Sigma_hat, rho, epsilon, k)

            lb, _ = lower_validity_bound(new_cfs, mu_hat, Sigma_hat, rho)
            diversity = compute_diversity(
                new_cfs, transformer.data_interface, weights=None)
            dpp = compute_dpp(new_cfs)
            prox = compute_proximity(test_ins, new_cfs)

            diversity_list.append(diversity)
            dpp_list.append(dpp)
            lb_list.append(lb)
            prox_list.append(prox)
            k_list.append(k)

        ret_k = {
            'idx': k_list,
            'lvb': lb_list,
            'dvs': diversity_list,
            'dpp': dpp_list,
            'prox': prox_list
        }

    # varying epsilon
    if vary_eps:
        logger.info("Run instance %d: varying eps", idx)
        k_list = []
        diversity_list = []
        dpp_list = []
        lb_list = []
        prox_list = []
        d = cfs.shape[1]
        rho = ec.rho
        eps_list = np.arange(ec.min_eps, ec.max_eps, ec.eps_step)
        k = int((num_cfs+1)/2)

        for epsilon in eps_list:
            new_cfs = mahalanobis_correction(
                cfs, mu_hat, Sigma_hat, rho, epsilon, k)

            lb, _ = lower_validity_bound(new_cfs, mu_hat, Sigma_hat, rho)
            diversity = compute_diversity(
                new_cfs, transformer.data_interface, weights=None)
            dpp = compute_dpp(new_cfs)
            prox = compute_proximity(test_ins, new_cfs)

            diversity_list.append(diversity)
            dpp_list.append(dpp)
            lb_list.append(lb)
            prox_list.append(prox)

        ret_eps = {
            'idx': eps_list,
            'lvb': lb_list,
            'dvs': diversity_list,
            'dpp': dpp_list,
            'prox': prox_list
        }

    logger.info("Done instance %d", idx)
    return ret_k, ret_eps


def __run_method_on_dataset(dname, ec, wdir, mname, cname,
                            num_proc=1, seed=None, logger=None):
    method = method_funcs[mname]
    classifier = classifier_classes[cname]

    sp = ec.ept1.synthetic_params
    params = (sp['sample_size'], sp['mean_0'], sp['cov_0'],
              sp['mean_1'], sp['cov_1'], sp['seed'])

    dataset, numerical = helpers.get_dataset(dname, params=params)

    dice_data = dice_ml.Data(dataframe=dataset,
                             continuous_features=numerical,
                             outcome_name='label')

    y = dataset['label']
    X = dataset.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    # Just in case the dataset is too small, 
    # use the instance in the training set as original input x_0 for generate counterfactuals
    X_test = X_test.append(X_train)
    y_test = y_test.append(y_train)

    full_dataset, full_numerical = helpers.get_full_dataset(
        dname, params=params)
    full_dice_data = dice_ml.Data(dataframe=full_dataset,
                                  continuous_features=full_numerical,
                                  outcome_name='label')
    # train a classifier
    transformer = DataTransformer(full_dice_data)
    clf = classifier(transformer)
    clf.train(X_train, y_train, max_iter=2000)

    theta0 = clf.weights

    dice_model = dice_ml.Model(
        model=clf.get_model(), backend=classifier.backend)

    if isinstance(clf, LRTorch):
        dice = DicePyTorchWrapper(full_dice_data, dice_model)
    else:
        dice = dice_ml.Dice(
            full_dice_data, dice_model, method=ec.ept1.dice_method)

    y_pred = clf.predict(X_test)
    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
    uds_X, uds_y = uds_X.head(
        ec.ept1.max_test_ins), uds_y.head(ec.ept1.max_test_ins)
    num_cfs = ec.ept1.num_cfs

    org_clfs = get_pretrain(cname, dname, ec.ept1.num_clfs)

    org_theta = np.stack(
        [org_clfs[i].weights for i in range(len(org_clfs))], axis=1)

    d = theta0.shape[0]
    # mu_hat = theta0
    # Sigma_hat = ec.ept1.beta * np.identity(d)
    mu_hat = np.mean(org_theta, axis=1)
    Sigma_hat = np.cov(org_theta)

    jobs_args = []
    for idx in range(len(uds_X)):
        test_ins_df = uds_X[idx:idx+1]
        jobs_args.append((
            idx, test_ins_df, method, clf, dice, transformer,
            mu_hat, Sigma_hat, ec.ept1, logger, True, True))

    # fix epsilon, varying k
    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(__run_test_ins)(
        *jobs_args[i]) for i in range(len(jobs_args)))

    metrics = ['idx', 'lvb', 'dvs', 'dpp', 'prox']
    data = [{}, {}]
    for i in range(len(data)):
        for m in metrics:
            data[i][m] = []

    for i in range(2):
        for m in metrics:
            for ret in rets:
                if ret[i] is not None:
                    data[i][m].append(np.array(ret[i][m]))
            data[i][m] = np.array(data[i][m])

    print(data)
    pdump(data, f'{dname}.pickle', wdir)



def __plot_2(datasets, wdir, ec):

    def __plot_dij(ax, bx, data, markers, colors):
        idx = np.mean(data['idx'], axis=0)
        lvb = np.mean(data['lvb'], axis=0)
        # dvs = np.mean(data['dvs'], axis=0)
        dvs = np.mean(data['dpp'], axis=0)
        prox = np.mean(data['prox'], axis=0)

        p1 = ax.plot(idx, lvb, color=colors[0], marker=markers[0])
        p2 = bx.plot(idx, prox, color=colors[1], marker=markers[1])
        # p3 = cx.plot(idx, prox, color=colors[2], marker=markers[2])
        ps = [p1[0], p2[0]]
        return ps

    plt.style.use('seaborn-white')
    nd = len(datasets)
    data = {}
    dname_map = {
        'synthesis': 'Synthetic data',
        'german': 'Correction shift',
        'sba': 'Temporal shift',
        'student': 'Geospatial shift'
    }
    for dname in datasets:
        data[dname] = pload(f'{dname}.pickle', wdir)

    metrics = ['lvb', 'dvs', 'prox']
    palette = plt.get_cmap('Set1')
    colors = [palette(1), palette(2), palette(4)]
    markers = ['o', '^', 'v']
    axis_label = ["$L^*$", "Proximity"]

    fsize = [0, (9, 5), (11, 7), (12, 9)]
    fig, axs = plt.subplots(nd, 2, figsize=fsize[nd], sharex='col', sharey='row')
    # fig, axs = plt.subplots(nd, 2, figsize=(12, fsize[nd]))
    fig.subplots_adjust(right=0.80)
    if nd == 1:
        axs = np.expand_dims(axs, axis=0)
    print(axs.shape)
    bxs = np.empty_like(axs, dtype=object)
    all_axs = [axs, bxs]

    for i in range(nd):
        for j in range(2):
            bxs[i, j] = axs[i, j].twinx()
            bxs[i, j].yaxis.label.set_color(colors[1])
            bxs[i, j].tick_params(axis='y', colors=colors[1])
            bxs[i, j].get_yaxis().get_major_formatter().set_useOffset(False)

        bxs[i, 0].spines["left"].set_position(("axes", -0.2))
        bxs[i, 0].spines["left"].set_visible(False)

        # bxs[i, 0].get_shared_y_axes().join(bxs[i, 0], bxs[i, 1])
        bxs[i, 0].set_yticklabels([])

    for k, xxs in enumerate(all_axs):
        for i in range(nd):
            for j in range(2):
                xxs[i, j].yaxis.label.set_color(colors[k])
                xxs[i, j].tick_params(axis='y', colors=colors[k])
                # for sp in xxs[i, j].spines.values():
                # sp.set_color(colors[k])
                # xxs[i, j].spines.set_color(colors[k])
                xxs[i, j].grid(b=False)
            axs[i, 0].set_ylabel(axis_label[0])
            bxs[i, 1].set_ylabel(axis_label[1])

    ps = None
    for i in range(nd):
        for j in range(2):
            dij = data[datasets[i]][j]
            ps = __plot_dij(axs[i, j], bxs[i, j],
                       dij, markers, colors)
        if nd != 1:
            bxs[i][0].set_ylabel(dname_map[datasets[i]])
        bxs[i][0].yaxis.set_label_coords(-0.23, 0.5)
        bxs[i][0].yaxis.label.set_color('black')

    axs[nd-1, 0].set_xlabel('$K$')
    axs[0, 0].set_title(f'$\Delta = {ec.ept1.epsilon}$, varying $K$')
    axs[0, 1].set_title(f'$K = {ec.ept1.num_corrections}$, varying $\Delta$')
    axs[nd-1, 1].set_xlabel('$\Delta$')

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, name in enumerate(axis_label):
        ax.plot([], marker=markers[i], label=name, color=colors[i])
    ax.legend(frameon=True,
               loc='upper center', bbox_to_anchor=(0., 1 + 0.2/nd, 1., 0.2/nd), ncol=2)
    ax.grid(b=False)
    plt.tight_layout(pad=4.0)
    dname = '_'.join(datasets)
    filename = os.path.join(wdir, f'{dname}_{ec.ept1.max_test_ins}_2.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    # plt.show()

    return 0

def __plot_1(datasets, wdir, ec):

    def __plot_dij(ax, bx, cx, data, markers, colors):
        idx = np.mean(data['idx'], axis=0)
        lvb = np.mean(data['lvb'], axis=0)
        # dvs = np.mean(data['dvs'], axis=0)
        dvs = np.mean(data['dpp'], axis=0)
        prox = np.mean(data['prox'], axis=0)

        p1 = ax.plot(idx, lvb, color=colors[0], marker=markers[0], alpha=0.8)
        p2 = bx.plot(idx, dvs, color=colors[1], marker=markers[1], alpha=0.8)
        p3 = cx.plot(idx, prox, color=colors[2], marker=markers[2], alpha=0.8)
        ps = [p1[0], p2[0], p3[0]]
        return ps

    plt.style.use('seaborn-white')
    nd = len(datasets)
    data = {}
    dname_map = {
        'synthesis': 'Synthetic data',
        'german': 'Correction shift',
        'sba': 'Temporal shift',
        'student': 'Geospatial shift'
    }
    for dname in datasets:
        data[dname] = pload(f'{dname}.pickle', wdir)

    metrics = ['lvb', 'dvs', 'prox']
    palette = plt.get_cmap('Set1')
    colors = [palette(1), palette(2), palette(4)]
    markers = ['o', '^', 'v']
    axis_label = ["$L^*$", "Diveristy", "Proximity"]

    fsize = [0, 5, 7, 9]
    fig, axs = plt.subplots(nd, 2, figsize=(
        11, fsize[nd]), sharex='col', sharey='row')
    fig.subplots_adjust(right=0.80)
    if nd == 1:
        axs = np.expand_dims(axs, axis=0)
    print(axs.shape)
    bxs = np.empty_like(axs, dtype=object)
    cxs = np.empty_like(axs, dtype=object)
    all_axs = [axs, bxs, cxs]

    for i in range(nd):
        for j in range(2):
            bxs[i, j] = axs[i, j].twinx()
            cxs[i, j] = axs[i, j].twinx()
            # bxs[i, j].spines["right"].set_position(("axes", 1.2))
            cxs[i, j].set_frame_on(True)
            cxs[i, j].patch.set_visible(False)

            bxs[i, j].yaxis.label.set_color(colors[1])
            bxs[i, j].tick_params(axis='y', colors=colors[1])
            cxs[i, j].yaxis.label.set_color(colors[2])
            cxs[i, j].tick_params(axis='y', colors=colors[2])
            cxs[i, j].spines["right"].set_visible(True)
            bxs[i, j].get_yaxis().get_major_formatter().set_useOffset(False)
            cxs[i, j].get_yaxis().get_major_formatter().set_useOffset(False)
            axs[i, j].get_yaxis().get_major_formatter().set_useOffset(False)

        cxs[i, 0].spines["right"].set_position(("axes", 1.05))
        cxs[i, 1].spines["right"].set_position(("axes", 1.2))
        bxs[i, 0].spines["left"].set_position(("axes", -0.15))
        bxs[i, 0].spines["left"].set_visible(False)

        bxs[i, 0].get_shared_y_axes().join(bxs[i, 0], bxs[i, 1])
        cxs[i, 0].get_shared_y_axes().join(cxs[i, 0], cxs[i, 1])
        bxs[i, 0].set_yticklabels([])
        cxs[i, 0].set_yticklabels([])

    for k, xxs in enumerate(all_axs):
        for i in range(nd):
            for j in range(2):
                xxs[i, j].yaxis.label.set_color(colors[k])
                xxs[i, j].tick_params(axis='y', colors=colors[k])
                # for sp in xxs[i, j].spines.values():
                # sp.set_color(colors[k])
                # xxs[i, j].spines.set_color(colors[k])
                xxs[i, j].grid(b=False)
            axs[i, 0].set_ylabel(axis_label[0])
            bxs[i, 1].set_ylabel(axis_label[1])
            cxs[i, 1].set_ylabel(axis_label[2])

    ps = None
    for i in range(nd):
        for j in range(2):
            dij = data[datasets[i]][j]
            ps = __plot_dij(axs[i, j], bxs[i, j], cxs[i, j],
                       dij, markers, colors)
        bxs[i][0].set_ylabel(dname_map[datasets[i]])
        bxs[i][0].yaxis.set_label_coords(-0.23, 0.5)
        bxs[i][0].yaxis.label.set_color('black')

    axs[nd-1, 0].set_xlabel('$K$')
    axs[0, 0].set_title(f'$\Delta = {ec.ept1.epsilon}$, varying $K$')
    axs[0, 1].set_title(f'$K = {ec.ept1.num_corrections}$, varying $\Delta$')
    axs[nd-1, 1].set_xlabel('$\Delta$')

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, name in enumerate(axis_label):
        ax.plot([], marker=markers[i], label=name, color=colors[i])
    ax.legend(frameon=True,
               loc='upper center', bbox_to_anchor=(0., 1 + 0.2/nd, 1., 0.2/nd), ncol=3)
    plt.tight_layout(pad=4.0)
    dname = '_'.join(datasets)
    filename = os.path.join(wdir, f'{dname}_{ec.ept1.max_test_ins}.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    # plt.show()

    return 0


def run_ept_1(ec, wdir, datasets, classifiers,
              num_proc=4, plot_only=False, seed=None, logger=None):
    logger.info("Running ept 1...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.ept1.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.ept1.all_classifiers

    mname = 'mahalanobis'

    for cname in classifiers:
        for dname in datasets:
            if not plot_only:
                __run_method_on_dataset(
                    dname, ec, wdir, mname,
                    cname, num_proc, seed, logger)

        __plot_1(datasets, wdir, ec)

    logger.info("Done ept 1.")
