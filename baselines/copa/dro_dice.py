import sys
sys.path.append('./copa/')

import torch
import numpy as np
import pandas as pd
from autograd import value_and_grad
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase

from utils_copa import sqrtm_psd, gelbrich_dist

class DroDicePGDT(ExplainerBase):

    def __init__(self, data_interface, model_interface,
                 mean_weights, cov_weights, robust_weight=10.0,
                 diversity_weight=3.0, lambd=0.7, zeta=1, max_iter=500,
                 epsilon=0.1, learning_rate=0.005,
                 features_to_vary='all', verbose=False, **kwargs):
        self.max_iter = max_iter
        self.diversity_weight = diversity_weight
        self.robust_weight = robust_weight
        self.num_stable_iter = 0
        self.learning_rate = learning_rate
        self.max_stable_iter = 3
        self.loss_diff_threshold = 1e-7
        self.transformer = None
        self.sigma_perturb_init = 1
        self.compute_value_and_grad = value_and_grad(self._compute_loss)
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.epsilon = epsilon
        self.lambd = lambd
        self.zeta = zeta
        self.verbose = verbose
        self.model_interface = model_interface

        super(DroDicePGDT, self).__init__(
            data_interface, model_interface, **kwargs)

        self.clf = model_interface.model
        self.features_to_vary = features_to_vary
        self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)
        self.sqrtm_cov_weights = torch.tensor(sqrtm_psd(cov_weights)).double()
        self.mean_weights_tensor = torch.tensor(mean_weights).double()
        self.cov_weights_tensor = torch.tensor(cov_weights).double()
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()

    def _cost_func(self, x, y):
        """_cost_func.
            cost function is L1 distance
        """
        return torch.norm(x-y, 2)

    def _compute_proximity_loss(self, test_ins, cfs):
        proximity = 0
        for i in range(len(cfs)):
            proximity += self._cost_func(test_ins, cfs[i])

        return proximity / len(cfs)

    def _compute_robust_loss(self, cfs):
        rs = []
        for i in range(len(cfs)):
            r = torch.dot(cfs[i], self.mean_weights_tensor) / \
                torch.norm(self.sqrtm_cov_weights @ cfs[i])
            rs.append(r)

        return -min(rs)

    def _compute_diversity_loss(self, cfs):
        diversity = 0
        num_cfs = len(cfs)
        for i in range(num_cfs):
            for j in range(i+1, num_cfs):
                diversity += cfs[i].T @ self.cov_weights_tensor @ cfs[j] / \
                    (torch.norm(self.sqrtm_cov_weights @ cfs[i]) *
                     torch.norm(self.sqrtm_cov_weights @ cfs[j]))
        return diversity / (num_cfs * (num_cfs - 1) / 2)

    def _compute_dpp_loss(self, cfs):
        num_cfs = len(cfs)
        det_entries = torch.ones((num_cfs, num_cfs))
        for i in range(num_cfs):
            for j in range(num_cfs):
                det_entries[(i, j)] = 1.0 / \
                    (1.0 + self._cost_func(cfs[i], cfs[j]))
                if i == j:
                    det_entries[(i, j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return -diversity_loss

    def _compute_loss(self, cfs):
        self.robust_loss = self._compute_robust_loss(cfs)
        self.proximity_loss = self._compute_proximity_loss(self.test_ins, cfs)
        # self.diversity_loss = self._compute_diversity_loss(cfs)
        self.diversity_loss = self._compute_dpp_loss(cfs)
        loss = self.proximity_loss + self.robust_weight * self.robust_loss + \
            self.diversity_weight * self.diversity_loss
        return loss

    def _get_output(self, cfs, w):
        with torch.no_grad():
            return torch.tensor([torch.dot(cf, w) for cf in cfs]).numpy()

    def print_cfs(self, cfs, grad=False):
        if grad:
            print(torch.stack([cf.grad.data for cf in cfs],
                  axis=0).detach().numpy())
        else:
            with torch.no_grad():
                print(torch.stack(cfs, axis=0).numpy())

    def _init_cfs(self, x_0, total_CFs):
        cfs = np.tile(x_0, (total_CFs, 1))
        cfs = cfs + np.random.randn(*cfs.shape) * self.sigma_perturb_init
        cfs[:, 0] = 1.0
        return cfs

    def _project(self, cfs):
        with torch.no_grad():
            w = self.mean_weights_tensor
            for i in range(len(cfs)):
                # cfs[i][0] = 1
                cfs[i][1:] = cfs[i][1:] - min(0, torch.dot(w, cfs[i]) - self.epsilon) \
                    * w[1:] / torch.norm(w[1:]) ** 2
        return cfs

    def _check_termination(self, loss_diff):
        # print(loss_diff, self.loss_diff_threshold, self.num_stable_iter)
        if loss_diff <= self.loss_diff_threshold:
            self.num_stable_iter += 1
            return (self.num_stable_iter >= self.max_stable_iter)
        else:
            self.num_stable_iter = 0
            return False

    def _generate_counterfactuals(self, query_instance, total_CFs,  **kwargs):
      pass


class DroDicePGDAD(DroDicePGDT):
    def _generate_counterfactuals(self, query_instance, total_CFs, **kwargs):
        
        test_ins = query_instance.squeeze()
        test_ins = np.concatenate((np.array([1]), test_ins))
        self.test_ins = torch.tensor(test_ins).double()
        initial_cfs = self._init_cfs(test_ins, total_CFs)
        num_cfs, d = initial_cfs.shape

        cfs = [torch.tensor(cf, requires_grad=True) for cf in initial_cfs]
        cfs = self._project(cfs)
        
        optim = torch.optim.Adam(cfs, self.learning_rate)

        loss_diff = 1.0
        prev_loss = 0.0
        self.num_stable_iter = 0
        
        for num_iter in range(self.max_iter):
            optim.zero_grad()

            loss_value = self._compute_loss(cfs)
            loss_value.backward()

            # freeze features other than feat_to_vary_idxs
            for ix in range(num_cfs):
                for jx in range(d):
                    if jx-1 not in self.feat_to_vary_idxs:
                        cfs[ix].grad[jx] = 0.0

            
            optim.step()
            cfs = self._project(cfs)
            

            if self.verbose:
  
              print("Iter %d: loss: %f" % (num_iter, loss_value.data.item()))

              print("---- Robust loss: %f * %f; Proximity loss: %f * %f; Diversity loss: %f * %f" %
                    (self.robust_weight, self.robust_loss,
                      1, self.proximity_loss,
                      self.diversity_weight, self.diversity_loss))

            # projection step
            for ix in range(num_cfs):
                for jx in range(1, d):
                    cfs[ix].data[jx] = torch.clamp(cfs[ix][jx],
                                                   min=self.minx[0][jx-1],
                                                   max=self.maxx[0][jx-1])

            # print(self._get_output(cfs, self.mean_weights_tensor))

            loss_diff = prev_loss - loss_value.data.item()
            if self._check_termination(loss_diff):
                break

            prev_loss = loss_value.data.item()

        
        
        col_names = self.data_interface.feature_names
        cfs = np.array([cf.detach().numpy() for cf in cfs])
        
        cfs_df = pd.DataFrame(cfs[:, 1:], columns=col_names)
        
        
        cfs_pred = self.clf.predict(cfs_df)
        cfs_df[self.data_interface.outcome_name] = cfs_pred
        return cfs_df