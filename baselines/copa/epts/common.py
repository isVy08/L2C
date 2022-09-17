import numpy as np
import torch
import random

# from classifiers import LRSklearn, LRTorch
from utils.utils import compute_diversity, compute_dpp, \
    compute_proximity, compute_validity
from validity import lower_validity_bound, upper_validity_bound
from correction import mahalanobis_correction
from dro_dice import DroDicePGDAD, DroDicePGDT


classifier_classes = {
    # 'lrs': LRSklearn,
    'lrt': LRTorch
}


def __dice_generator(idx, test_ins_df, num_cfs, ec, dice, mu_hat, Sigma_hat,
                     transformer, shifted_clfs, seed, logger,
                     params=None, eval_cfs_func=None, eval_args=None):
    logger.info("Running method: Dice on test instance %d", idx)
    np.random.seed(seed + 3)
    torch.manual_seed(seed + 4)
    random.seed(seed + 5)

    cfps = dice.generate_counterfactuals(test_ins_df, total_CFs=num_cfs,
                                         desired_class="opposite",
                                         posthoc_sparsity_param=None)

    cfp = cfps.cf_examples_list[0]

    test_ins = cfp.test_instance_df
    cf_plan = cfp.final_cfs_df
    cfs = cfp.final_cfs_df_sparse
    cfs = np.hstack([np.ones((num_cfs, 1)), cfs])

    # transforming categorical features if exist
    test_ins = transformer.transform(test_ins.drop('label', axis=1),
                                     intercept_feature=True, tonumpy=True).squeeze()

    d = cfs.shape[1]
    rho = ec.rho
    epsilon = ec.epsilon

    if eval_cfs_func is not None:
        return eval_cfs_func(cfs, eval_args)
    else:
        diversity = compute_diversity(
            cfs, transformer.data_interface, weights=None)
        lb, _ = lower_validity_bound(cfs, mu_hat, 1.0001 *Sigma_hat, ec.rho)
        dpp = compute_dpp(cfs)
        prox = compute_proximity(test_ins, cfs)
        validity = compute_validity(cfs, shifted_clfs)
        current_validity = compute_validity(cfs, [dice.model.model])

        logger.info("=> Done: Dice on test instance %d", idx)
        return validity, lb, diversity, dpp, prox, current_validity


def _project(cfs, w, epsilon):
    for i in range(len(cfs)):
        cfs[i][0] = 1
        cfs[i][1:] = cfs[i][1:] - min(0, np.dot(w, cfs[i]) - epsilon) \
            * w[1:] / np.linalg.norm(w[1:]) ** 2
    return cfs


def __mahalanobis_drodice_generator(idx, test_ins_df, num_cfs, ec, dice, mu_hat, Sigma_hat,
                                    transformer, shifted_clfs, seed, logger,
                                    params=None, eval_cfs_func=None, eval_args=None):
    logger.info("Running method: Mahalanobis on test instance %d", idx)
    np.random.seed(seed + 3)
    torch.manual_seed(seed + 4)
    random.seed(seed + 5)

    cfps = dice.generate_counterfactuals(test_ins_df, total_CFs=num_cfs,
                                         desired_class="opposite",
                                         posthoc_sparsity_param=None)

    cfp = cfps.cf_examples_list[0]

    test_ins = cfp.test_instance_df
    cf_plan = cfp.final_cfs_df
    cfs = cfp.final_cfs_df_sparse
    cfs = np.hstack([np.ones((num_cfs, 1)), cfs])

    # transforming categorical features if exist
    test_ins = transformer.transform(test_ins.drop('label', axis=1),
                                     intercept_feature=True, tonumpy=True).squeeze()

    rho = ec.rho
    epsilon = ec.epsilon
    k = int(ec.num_corrections)

    cfs = _project(cfs, mu_hat, ec.epsilon_pgd)

    cfs = mahalanobis_correction(
        cfs, mu_hat, Sigma_hat, rho, epsilon, k)

    if eval_cfs_func is not None:
        return eval_cfs_func(cfs, eval_args)
    else:
        diversity = compute_diversity(
            cfs, transformer.data_interface, weights=None)
        lb, _ = lower_validity_bound(cfs, mu_hat, 1.0001 *Sigma_hat, ec.rho)
        dpp = compute_dpp(cfs)
        prox = compute_proximity(test_ins, cfs)
        validity = compute_validity(cfs, shifted_clfs)
        current_validity = compute_validity(cfs, [dice.model.model])

        logger.info("=> Done: Mahalanobis on test instance %d", idx)
        return validity, lb, diversity, dpp, prox, current_validity


def __drodice_pgd_generator(idx, test_ins_df, num_cfs, ec, dice, mu_hat, Sigma_hat,
                            transformer, shifted_clfs, seed, logger,
                            params=None, eval_cfs_func=None, eval_args=None):
    logger.info(
        "Running method: projected gradient descent on test instance %d", idx)
    np.random.seed(seed + 3)
    torch.manual_seed(seed + 4)
    random.seed(seed + 5)

    mname = params['mname']
    robust_weight = ec.params[mname]['robust_weight']
    diversity_weight = ec.params[mname]['diversity_weight']

    dro_dice = DroDicePGDAD(dice.data_interface, dice.model,
                            mean_weights=mu_hat, cov_weights=Sigma_hat,
                            max_iter=ec.max_iter, learning_rate=ec.learning_rate,
                            robust_weight=robust_weight, diversity_weight=diversity_weight,
                            lambd=ec.lambd, zeta=ec.zeta, epsilon=ec.epsilon_pgd, verbose=False)

    cfps = dro_dice.generate_counterfactuals(test_ins_df, total_CFs=num_cfs,
                                             desired_class="opposite")
    cfp = cfps.cf_examples_list[0]

    test_ins = cfp.test_instance_df
    cf_plan = cfp.final_cfs_df
    cfs = cfp.final_cfs_df_sparse

    # transforming categorical features if exist
    test_ins = transformer.transform(test_ins.drop('label', axis=1),
                                     intercept_feature=True, tonumpy=True).squeeze()

    d = cfs.shape[1]

    if eval_cfs_func is not None:
        return eval_cfs_func(cfs, eval_args)
    else:
        diversity = compute_diversity(
            cfs, transformer.data_interface, weights=None)
        # 1.0001 is to prevent error
        lb, _ = lower_validity_bound(cfs, mu_hat, 1.0001 * Sigma_hat, ec.rho)
        dpp = compute_dpp(cfs)
        prox = compute_proximity(test_ins, cfs)
        validity = compute_validity(cfs, shifted_clfs)
        current_validity = compute_validity(cfs, [dice.model.model])

        logger.info(
            "=> Done: projected gradient descent on test instance %d", idx)
        return validity, lb, diversity, dpp, prox, current_validity


method_funcs = {
    'dice': __dice_generator,
    'mahalanobis': __mahalanobis_drodice_generator,
    'pgd': __drodice_pgd_generator,
    'pgd1': __drodice_pgd_generator,
    'pgd2': __drodice_pgd_generator,
}
