import sys
from utils import *
from utils_eval import *
from data_loader import *

from tqdm import tqdm

import baselines.dice_ml as dice_ml
from baselines.dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from baselines.copa.dro_dice import DroDicePGDAD, DroDicePGDT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def generate_baseline(X_test, dg, cls, explainer, method, num_samples, max_iters=10000):

    N = X_test.shape[0]
    start = time.time()
    features_to_vary = [
        col for col in X_test.columns if col not in dg.immutable_cols]
    finals = []
    for i in tqdm(range(N)):

        x = X_test.iloc[i:i+1, ]

        if method in ('dice', 'copa'):
            output = explainer._generate_counterfactuals(x, total_CFs=num_samples, desired_class="opposite",
                                                         features_to_vary=features_to_vary,
                                                         posthoc_sparsity_algorithm="linear",
                                                         limit_steps_ls=max_iters,
                                                         )
            if method == 'dice':
                output = output.final_cfs_df

        elif method == 'fvae':
            output = explainer.generate_counterfactuals(
                x, num_samples, "opposite", max_iters)

        elif method == 'certifai':
            fixed = [X_test.columns.tolist().index(col)
                     for col in dg.immutable_cols]
            explainer.fit(cls, x, pytorch=False, final_k=num_samples, generations=max_iters,
                          fixed=fixed)
            cfs = explainer.results[0][1]
            output = pd.DataFrame(cfs, columns=dg.encoded_columns)

        else:
            raise ValueError('Method not supported')

        finals.append(output)

    end = time.time()
    finals = pd.concat(finals)
    print('\n', end-start, '\n')

    return finals

def evaluate_baseline(X_test, synth_df, num_samples, dg, cls):  
    N = synth_df.shape[0]
    i = 0 
    SP, DI, VA, CG, HM = 0, 0, 0, 0, 0
    SELF_CAU = []
    for j in tqdm(range(0, N, num_samples)): 

      
      x = X_test.iloc[i:i+1, ]
      x = x.to_numpy()

      s0, _ = parse_sample(dg, x)
      y = cls.predict(x)[0]
      s0[dg.target_col] =  y

      if isinstance(synth_df, pd.DataFrame):
        output = synth_df.iloc[j:j+num_samples, :]
      else:
        output = synth_df[j:j+num_samples, :]
        output = pd.DataFrame(output, columns = X_test.columns.tolist() + [dg.target_col])

        
      samples, vac = get_clean_samples(output, dg, cls)
      
      sp, di, va = evaluate(dg, num_samples, s0, samples, False)
      

      if va > 0:
        CG += 1
      
      DI += di 
      VA += va
      SP += sp

      
      scau = check_causal_relations(dg, s0, samples)
      
      if scau > -1.0:
        SELF_CAU.append(scau)
    
      i += 1

    N = N / num_samples
    
    SPARSITY = SP / N
    DIVERSITY = DI / N
    HM = (2 * DIVERSITY * (1 - SPARSITY)) / (DIVERSITY  + 1 - SPARSITY)
    print('Sparsity - Diveristy - HarMean - Validity - Coverage - Self Cau') # sparsity is negative in this sense
    return SPARSITY, DIVERSITY, HM, VA/N, CG/N, np.mean(SELF_CAU)


if __name__ == '__main__':

    method = sys.argv[1]
    name = sys.argv[2]

    dg, immutable_cols = load_data(name, False, device, 'default')
    classifiers = load_blackbox(name, dg, False)

    X_train, X_val, X_test, y_train, y_val, y_test = dg.transform(
        return_tensor=False)
    X_train[dg.target_col] = y_train


    
    num_samples = 100
    max_iter = 10000  # Adjust max iteration based on your time budget!

    if method in ('dice', 'copa'):
        
        dice_data = dice_ml.Data(dataframe=X_train, continuous_features=dg.num_cols, outcome_name=dg.target_col)
        dice_data.set_continuous_feature_indexes(X_test)

        if method == 'copa':
            dice_data.create_ohe_params()
            weight_list = []
            for cls_index in range(5):
                weights = np.hstack(
                    (classifiers[cls_index].intercept_, classifiers[cls_index].coef_.squeeze()))
                weight_list.append(weights)

            org_theta = np.stack(weight_list, axis=1)
            mu_hat = np.mean(org_theta, axis=1)
            Sigma_hat = 1.5 * np.cov(org_theta) + 1e+5

        num_samples = 100

        for cls_index in range(5):
            cls = classifiers[cls_index]
            dice_model = dice_ml.Model(model=cls, backend='sklearn')

            if method == 'dice':
                explainer = dice_ml.Dice(
                    dice_data, dice_model, method='random')

            elif method == 'copa':
                explainer = DroDicePGDAD(dice_data, dice_model, mean_weights=mu_hat,
                                         cov_weights=Sigma_hat, verbose=False, max_iter=max_iter)

            finals = generate_baseline(
                X_test, dg, cls, explainer, method, num_samples, max_iter)
            print(
                f'Start evaluating {method} at black-box {cls_index + 1} ...')
            evaluate_baseline(X_test, finals, num_samples, dg, cls)

    elif method == 'fvae':

        dice_data = dice_ml.Data(
            dataframe=X_train, continuous_features=dg.encoded_columns, outcome_name=dg.target_col)
        backend = {'model': 'base_model.BaseModel',
                'explainer': 'feasible_base_vae.FeasibleBaseVAE'}

        params = {'german': [10, 42], 'admission': [
            30, 62], 'sba': [70, 42], 'student': [10, 62]}
        
        for cls_index in range(5):
            cls = classifiers[cls_index]
            if name in ('german', 'student'):
                ml_model = LinearClassifier(cls, device)
                dice_model = dice_ml.Model(model=ml_model, backend=backend)
            else:
                dice_model = dice_ml.Model(model=cls, backend=backend)

            explainer = dice_ml.Dice(
                dice_data, dice_model, encoded_size=params[name][0], validity_reg=params[name][1], margin=0.165, epochs=500)
            explainer.save_path = f'./model/{name}/VAE_{cls_index + 1}.pt'
            if os.path.isfile(explainer.save_path):
                explainer.train(pre_trained=1)
            else:
                explainer.train(pre_trained=0)
            explainer.cf_vae.to(device)

            finals = generate_baseline(
                X_test, dg, cls, explainer, 'FVAE', num_samples, max_iter)
            print(
                f'Start evaluating {method} at black-box {cls_index + 1} ...')
            evaluate_baseline(X_test, finals, num_samples, dg, cls)

    elif method == 'mcce':
        from baselines.mcce import *
        feature_order = dg.num_cols + dg.cat_cols + [dg.target_col]

        def run_mcce(X_test, dg, cls, num_samples, backend='sklearn'):
            # Construct dataset object
            fixed_features = dg.immutable_cols
            feature_order = dg.num_cols + dg.cat_cols + [dg.target_col]
            dtypes = {col: 'category' for col in dg.cat_cols + [dg.target_col]}
            for col in dg.num_cols:
                dtypes[col] = 'float'
            dataset = Data(dg.df.iloc[:dg.train_size, ], feature_order,
                           dtypes, dg.target_col,  fixed_features, "OneHot", None)

            # Fit data
            for col in dg.encoded_columns:
                if col not in dtypes:
                  dtypes[col] = 'category'

            mcce = MCCE(fixed_features=dataset.fixed_features, fixed_features_encoded=dataset.fixed_features_encoded,
                        continuous=dataset.continuous, categorical=dataset.categorical, model=cls)
            mcce.fit(dataset.df, dtypes)
            synth_df = mcce.generate(X_test, num_samples)
            return synth_df

        for cls_index in range(5):
            start = time.time()
            cls = classifiers[cls_index]
            finals = run_mcce(
                X_test, dg, classifiers[cls_index], num_samples, 'sklearn')
            print(
                f'Start evaluating {method} at black-box {cls_index + 1} ...')
            evaluate_baseline(X_test, finals, num_samples, dg, cls)
 
    elif method == 'fastar':
      data_path = f'FastAR/fastar/datasets/my_{name}_data.pickle'
      print('Transfering datasets ...')
      packs = (X_train, X_val, X_test, dg.encoded_columns, dg.immutable_cols)
      write_pickle(packs, data_path)


      for i in range(1, 6):

          path = f'models/{name.upper()}_{i}'
          if name in ('german', 'student'):
              path = path + '.pickle'
          else:
              path = path + '.pt'
          
          print('Transfering models ...')
          if name in ('german', 'student'):
              os.system(
                  f"cp model/{name}/{name.upper()}_{i}.pickle FastAR/fastar/models/")
          else:
              os.system(f"cp model/{name}/{name.upper()}_{i}.pt FastAR/fastar/models/")     