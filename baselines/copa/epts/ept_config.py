from epts.config import Config


class Ept1(Config):

    __dictpath__ = 'ec.ept1'

    dice_method = 'PYT'

    all_methods = ['mahalanobis', 'gradient', 'dice', 'pgd']
    all_datasets = ['german', 'sba', 'student']
    all_classifiers = ['lrt']

    all_methods = ['heuristic', 'gradient']
    all_datasets = ['synthetic_data', 'adult_data']

    synthetic_params = {
        "sample_size": 1000,
        "mean_0": [-2, -2],
        "cov_0": [[1, 0], [0, 1]],
        "mean_1": [2, 2],
        "cov_1": [[1, 0], [0, 1]],
        "seed": 42
    }

    # ccommon
    max_test_ins = 100
    num_cfs = 5
    num_clfs = 1000
    num_corrections = (num_cfs+1)//2
    rho = 0.01
    epsilon = 0.1
    beta = 0.01

    # for varying eps
    min_eps = 0.05
    max_eps = 0.4
    eps_step = 0.05
    epsilon_pgd = 0.1



class Ept2(Config):
    __dictpath__ = 'ec.ept2'

    dice_method = 'PYT'

    all_methods = ['mahalanobis', 'gradient', 'dice', 'pgd']
    all_datasets = ['german', 'sba', 'student']
    all_classifiers = ['lrt']

    synthetic_params = {
        "sample_size": 1000,
        "mean_0": [-2, -2],
        "cov_0": [[3, 0], [0, 3]],
        "mean_1": [2, 2],
        "cov_1": [[1, 0], [0, 1]],
        "seed": 42
    }

    shifted_synthetic_params = {
        "sample_size": 1000,
        "mean_0": [-4, -4],
        "cov_0": [[1, 0], [0, 1]],
        "mean_1": [2, 2],
        "cov_1": [[1, 0], [0, 1]],
        "seed": 42
    }

    max_test_ins = 100
    num_cfs = 5
    num_clfs = 1000

    # parameters for DroDice
    num_corrections = (num_cfs+1)/2
    rho = 0.01
    epsilon = 0.1
    beta = 0.01

    params = {
        "synthesis": {
            "robust_weight": 1.0,
            "diversity_weight": 500.0,
        },
        "german": {
            "robust_weight": 2.0,
            "diversity_weight": 50.0,
        },
        "sba": {
            "robust_weight": 2.0,
            "diversity_weight": 10.0,
        },
        "student": {
            "robust_weight": 2.0,
            "diversity_weight": 50.0,
        },
        "pgd": {
            "robust_weight": 0.2,
            "diversity_weight": 2.0,
        },
        "pgd1": {
            "robust_weight": 0.5,
            "diversity_weight": 5.0,
        },
        "pgd2": {
            "robust_weight": 1.0,
            "diversity_weight": 10.0,
        },
    }

    # parameters for DroDiceGD
    learning_rate = 0.005
    robust_weight = 5.0
    diversity_weight = 50.0
    lambd = 0.7
    zeta = 1.0
    barrier_weight = 1.0
    epsilon_pgd = 0.1
    max_iter = 1000

class Ept3(Config):
    __dictpath__ = "ec.ept3"

    dice_method = 'PYT'

    all_methods = ['mahalanobis', 'gradient', 'dice', 'pgd']
    all_datasets = ['german', 'sba', 'student']
    all_classifiers = ['lrt']

    all_methods = ['heuristic', 'gradient']
    all_datasets = ['synthetic_data', 'adult_data']

    synthetic_params = {
        "sample_size": 1000,
        "mean_0": [-2, -2],
        "cov_0": [[1, 0], [0, 1]],
        "mean_1": [2, 2],
        "cov_1": [[1, 0], [0, 1]],
        "seed": 42
    }

    # ccommon
    max_test_ins = 100
    num_cfs = 5
    num_clfs = 1000
    num_corrections = (num_cfs+1)//2
    rho = 0.01
    epsilon = 0.1
    beta = 0.5

    # for varying eps
    min_eps = 0.05
    max_eps = 0.4
    eps_step = 0.05
    epsilon_pgd = 0.1

    params = {
        "pgd": {
            "robust_weight": 2.0,
            "diversity_weight": 200.0,
        },
    }

    # parameters for DroDiceGD
    Sigma_hat_beta = 0.5
    learning_rate = 0.005
    robust_weight = 5.0
    diversity_weight = 50.0
    lambd = 0.7
    zeta = 1.0
    barrier_weight = 1.0
    epsilon_pgd = 0.1
    max_iter = 1000

    sample_size = 1000
    min_alpha = 0
    max_alpha = 1.0
    num_alpha = 11

    min_beta = 0
    max_beta = 3
    num_beta = 11


class PretrainConfig(Config):
    __dictpath__ = "ec.pretrain"

    all_datasets = ['synthesis', 'german', 'sba', 'student']
    all_classifiers = ['logistic']

    synthetic_params = {
        "sample_size": 1000,
        "mean_0": [-2, -2],
        "cov_0": [[2, 0], [0, 2]],
        "mean_1": [2, 2],
        "cov_1": [[2, 0], [0, 2]],
        "seed": 42
    }

    shifted_synthetic_params = {
        "sample_size": 1000,
        "mean_0": [-4, -4],
        "cov_0": [[2, 0], [0, 2]],
        "mean_1": [2, 2],
        "cov_1": [[2, 0], [0, 2]],
        "seed": 42
    }

    num_classifiers = 1000


class EptConfig(Config):
    __dictpath__ = 'ec'
    __doc__ = 'Similar to run 9, change num epochs from 1000 to 1500'

    ept1 = Ept1
    ept2 = Ept2
    ept3 = Ept3
    pretrain = PretrainConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    if args.load is not None:
        EptConfig.from_file(args.load)
    EptConfig.to_file(args.dump, mode=args.mode)
