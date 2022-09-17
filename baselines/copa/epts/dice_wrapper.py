import dice_ml
from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import torch

import numpy as np
import random
import timeit
import copy

from dice_ml import diverse_counterfactuals as exp
from dice_ml.counterfactual_explanations import CounterfactualExplanations


class DicePyTorchWrapper(DicePyTorch):

    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter,
                             max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
                             init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
                             posthoc_sparsity_algorithm):
        """Finds counterfactuals by gradient-descent."""

        # Prepares user defined query_instance for DiCE.
        # query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encoding='one-hot')
        # query_instance = query_instance.iloc[0].values
        query_instance = self.data_interface.get_ohe_min_max_normalized_data(
            query_instance).iloc[0].values
        self.x1 = torch.tensor(query_instance)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(torch.tensor(query_instance).float())[0]
        if desired_class == "opposite":
            desired_class = 1.0 - np.round(test_pred)
        self.target_cf_class = torch.tensor(desired_class).float()

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

        # variables to backup best known CFs so far in the optimization process -
        # if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.min_dist_from_threshold = [100]*loop_find_CFs  # for backup CFs

        for loop_ix in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                self.initialize_CFs(query_instance, False)
            else:
                self.initialize_CFs(query_instance, init_near_query_instance)

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()

                # get loss and backpropogate
                loss_value = self.compute_loss()
                self.loss.backward()

                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        if jx not in self.feat_to_vary_idxs:
                            self.cfs[ix].grad[jx] = 0.0

                # update the variables
                self.optimizer.step()

                # projection step
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        self.cfs[ix].data[jx] = torch.clamp(
                            self.cfs[ix][jx], min=self.minx[0][jx], max=self.maxx[0][jx])

                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                temp_cfs_stored = self.round_off_cfs(assign=False)
                test_preds_stored = [self.predict_fn(
                    cf) for cf in temp_cfs_stored]

                if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or
                   (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    avg_preds_dist = np.mean(
                        [abs(pred[0]-self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                        self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                        for ix in range(self.total_CFs):
                            self.best_backup_cfs[loop_ix +
                                                 ix] = copy.deepcopy(temp_cfs_stored[ix])
                            self.best_backup_cfs_preds[loop_ix +
                                                       ix] = copy.deepcopy(test_preds_stored[ix])

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].detach().clone().numpy()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        if((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in self.cfs_preds))):
            for loop_ix in range(loop_find_CFs):
                if self.min_dist_from_threshold[loop_ix] != 100:
                    for ix in range(self.total_CFs):
                        self.final_cfs[loop_ix +
                                       ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                        self.cfs_preds[loop_ix+ix] = copy.deepcopy(
                            self.best_backup_cfs_preds[loop_ix+ix])

        # convert to the format that is consistent with dice_tensorflow
        query_instance = np.array([query_instance], dtype=np.float32)
        for tix in range(max(loop_find_CFs, self.total_CFs)):
            self.final_cfs[tix] = np.array(
                [self.final_cfs[tix]], dtype=np.float32)
            self.cfs_preds[tix] = np.array(
                [self.cfs_preds[tix]], dtype=np.float32)

            # if self.final_cfs_sparse is not None:
            #     self.final_cfs_sparse[tix] = np.array([self.final_cfs_sparse[tix]], dtype=np.float32)
            #     self.cfs_preds_sparse[tix] = np.array([self.cfs_preds_sparse[tix]], dtype=np.float32)
            #
            # checking if CFs are backed
            if isinstance(self.best_backup_cfs[0], np.ndarray):
                self.best_backup_cfs[tix] = np.array(
                    [self.best_backup_cfs[tix]], dtype=np.float32)
                self.best_backup_cfs_preds[tix] = np.array(
                    [self.best_backup_cfs_preds[tix]], dtype=np.float32)

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0]
                       for i in range(len(self.final_cfs))])
        raw_cfs = np.copy(cfs)
        # print(raw_cfs)
        final_cfs_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(
            cfs)
        cfs_preds = [np.round(preds.flatten().tolist(), 3)
                     for preds in self.cfs_preds]
        # print(cfs_preds)
        cfs_preds = [item for sublist in cfs_preds for item in sublist]
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)

        test_instance_df = self.data_interface.get_inverse_ohe_min_max_normalized_data(
            query_instance)
        test_instance_df[self.data_interface.outcome_name] = np.array(
            np.round(test_pred, 3))

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(
                final_cfs_df_sparse, test_instance_df, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            final_cfs_df_sparse = None

        m, s = divmod(self.elapsed, 60)
        if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))):
            self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
            # indexes of valid CFs
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            valid_ix = []  # indexes of valid CFs
            # print(self.target_cf_class, self.cfs_preds)
            for cf_ix, pred in enumerate(self.cfs_preds):
                if((self.target_cf_class == 0 and pred[0][0] < self.stopping_threshold) or
                   (self.target_cf_class == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1
                    valid_ix.append(cf_ix)

            if self.total_CFs_found == 0:
                print('No Counterfactuals found for the given configuation, ',
                      'perhaps try with different values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                      ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                      ' values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(
                drop=True)

        final_cfs_df_sparse = raw_cfs
        # print(raw_cfs)
        # returning only valid CFs
        return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse

    def _generate_counterfactuals(**kwargs):
        return 0
