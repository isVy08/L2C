import re
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class Data():

    def __init__(self, \
        df, 
        feature_order, 
        dtypes, 
        response, 
        fixed_features,  # these are the original feature names
        encoding_method="OneHot_drop_first",
        scaling_method="MinMax"):
        
        # df = pd.read_csv(path, header=0, names=feature_order) # assume no index column
        self.fixed_features = fixed_features
        self.target = response
        self.feature_order = feature_order
        self.dtypes = dtypes
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method

        self.categorical = [feat for feat in self.dtypes.keys() if dtypes[feat] == 'category']
        self.categorical.remove(self.target)
        self.continuous = [feat for feat in self.dtypes.keys() if dtypes[feat] != 'category']
        self.features = self.categorical + self.continuous
        self.cols = self.features + [self.target]
        
        df = df.astype(self.dtypes)

        # Convert target to 1 if "desirable" label; 0 otherwise
        # level = df[self.target].value_counts().idxmax()
        # df[self.target] = np.where(df[self.target] != level, 1, 0)
        
        self.df = df
        self.df_raw = df 
        # Fit scaler and encoder
        
        # self.scaler = self.fit_scaler(self.df[self.continuous], scaling_method)
        
        self.encoder = self.fit_encoder(self.df[self.categorical], encoding_method)
        self._identity_encoding = (encoding_method is None or encoding_method == "Identity")

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self.df = self.transform(self.df)

        # Can we get the fixed feature names after the transformation?
        self.categorical_encoded = self.encoder.get_feature_names(self.categorical).tolist()
        

        fixed_features_encoded = []
        for fixed in fixed_features:
            if fixed in self.categorical:
                for new_col in self.categorical_encoded:
                    match = re.search(fixed, new_col)
                    if match:
                        fixed_features_encoded.append(new_col)
            else:
                fixed_features_encoded.append(fixed)

        # print(type(fixed_features_encoded))
        self.fixed_features_encoded = fixed_features_encoded
        

    def transform(self, df):
        
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def inverse_transform(self, df):
        
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
            output = trans_function(output)

        return output

    def get_pipeline_element(self, key):
        
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    def __init_pipeline(self):
        return [
            # ("scaler", lambda x: self.scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: self.encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self):
        return [
            ("encoder", lambda x: self.decode(self.encoder, self.categorical, x)),
            # ("scaler", lambda x: self.descale(self.scaler, self.continuous, x)),
        ]
        
    def fit_encoder(self, df, encoding_method="OneHot_drop_first"):
        
        if encoding_method == "OneHot":
            encoder = preprocessing.OneHotEncoder(
                handle_unknown="error", sparse=False
            ).fit(df)
        elif encoding_method == "OneHot_drop_binary":
            encoder = preprocessing.OneHotEncoder(
                drop="if_binary", handle_unknown="error", sparse=False
            ).fit(df)
        elif encoding_method == "OneHot_drop_first":
            encoder = preprocessing.OneHotEncoder(
                drop="first", sparse=False
            ).fit(df)
        elif encoding_method is None or "Identity":
            encoder = preprocessing.FunctionTransformer(func=None, inverse_func=None)
            
        else:
            raise ValueError("Encoding Method not known")

        return encoder


    def fit_scaler(self, df, scaling_method="MinMax"):
        
        if scaling_method == "MinMax":
            scaler = preprocessing.MinMaxScaler().fit(df)
        elif scaling_method == "Standard":
            scaler = preprocessing.StandardScaler().fit(df)
        elif scaling_method is None or "Identity":
            scaler = preprocessing.FunctionTransformer(func=None, inverse_func=None)
        else:
            raise ValueError("Scaling Method not known")
        
        # X_scaled = self.scaler.transform(df)
        # X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

        return scaler # X_scaled[cols]


    def scale(self, fitted_scaler, features, df):
        
        output = df.copy()
        output[features] = fitted_scaler.transform(output[features])

        return output


    def descale(self, fitted_scaler, features, df) -> pd.DataFrame:
        
        output = df.copy()
        output[features] = fitted_scaler.inverse_transform(output[features])

        return output


    def encode(self, fitted_encoder, features, df):
        
        output = df.copy()
        encoded_features = fitted_encoder.get_feature_names(features)
        output[encoded_features] = fitted_encoder.transform(output[features])
        output = output.drop(features, axis=1)

        return output


    def decode(self, fitted_encoder, features, df):
        
        output = df.copy()
        encoded_features = fitted_encoder.get_feature_names(features)

        # Prevent errors for datasets without categorical data
        # inverse_transform cannot handle these cases
        if len(encoded_features) == 0:
            return output

        output[features] = fitted_encoder.inverse_transform(output[encoded_features])
        output = output.drop(encoded_features, axis=1)

        return output
    


    def fit_model(self, X, y, test_size=0.33):
        self.test_size = test_size

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = RandomForestClassifier(max_depth=None, random_state=0)
        self.model = clf.fit(X_train, y_train)
        
        pred_train = self.model.predict(X_train)
        pred_test = self.model.predict(X_test)

        fpr, tpr, _ = metrics.roc_curve(y_train, pred_train, pos_label=1)
        self.train_auc = metrics.auc(fpr, tpr)

        fpr, tpr, _ = metrics.roc_curve(y_test, pred_test, pos_label=1)
        self.test_auc = metrics.auc(fpr, tpr)

        self.model_prediction = clf.predict(X)

from sklearn.neighbors import NearestNeighbors


def distance(counterfactuals_without_nans, factual_without_nans, ml_model):
    
    arr_f = ml_model.get_ordered_features(factual_without_nans).to_numpy()
    arr_cf = ml_model.get_ordered_features(counterfactuals_without_nans).to_numpy()

    delta = arr_f - arr_cf 

    d1 = np.sum(np.invert(np.isclose(delta, np.zeros_like(delta))), axis=1, dtype=np.float).tolist()
    d1_old = np.sum(delta.round(2) != 0, axis=1, dtype=np.float).tolist()

    d2 = np.sum(np.abs(delta), axis=1, dtype=np.float).tolist()
    d3 = np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()

    return({'L0': d1, 'L1': d2, 'L2': d3})



def feasibility(
    counterfactuals_without_nans,
    factual_without_nans,
    cols,
    ):
    
    nbrs = NearestNeighbors(n_neighbors=5).fit(factual_without_nans[cols].values)

    results = []
    for i, row in counterfactuals_without_nans[cols].iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), 5, return_distance=True)[0]
        
        results.append(np.mean(knn))

    return results


def constraint_violation(
    df_decoded_cfs,
    df_factuals,
    continuous,
    categorical,
    fixed_features,
    ):
    
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))


    cfs_continuous_immutable = df_decoded_cfs[
        intersection(continuous, fixed_features)
    ]
    factual_continuous_immutable = df_factuals[
        intersection(continuous, fixed_features)
    ]

    continuous_violations = np.invert(
        np.isclose(cfs_continuous_immutable, factual_continuous_immutable)
    )
    continuous_violations = np.sum(continuous_violations, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    # check categorical by boolean comparison
    cfs_categorical_immutable = df_decoded_cfs[
        intersection(categorical, fixed_features)
    ]
    factual_categorical_immutable = df_factuals[
        intersection(categorical, fixed_features)
    ]

    categorical_violations = cfs_categorical_immutable != factual_categorical_immutable
    categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    return (continuous_violations + categorical_violations)


def success_rate(counterfactuals_without_nans, ml_model, cutoff=0.5):
    
    preds = ml_model.predict_proba(counterfactuals_without_nans)[:, [1]]
    preds = preds >= cutoff
    # {'success': preds>=cutoff, 'prediction': preds}
    return ([int(x) for x in preds])

from abc import ABC, abstractmethod

NUM_COLS_DTYPES = ['int', 'float', 'datetime', 'float64']
CAT_COLS_DTYPES = ['category', 'bool']


class Method(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def prepare_dfs(self, X_df, y_df=None, normalise_num_cols=False, one_hot_cat_cols=True, fit=True):
        
        X_df = X_df.copy()

        if y_df is not None and self.dtype in NUM_COLS_DTYPES:
          
            y_df = y_df.copy()

            not_nan_indices = y_df.notna()
            X_df = X_df.loc[not_nan_indices]
            y_df = y_df.loc[not_nan_indices]        

        if normalise_num_cols:
            if fit:
                num_cols = X_df.select_dtypes(NUM_COLS_DTYPES).columns.to_list()
                self.num_cols_range = {}
                for col in num_cols:
                    self.num_cols_range[col] = {'min': np.min(X_df[col]), 'max': np.max(X_df[col])}
                    X_df[col] = (X_df[col] - self.num_cols_range[col]['min']) / (self.num_cols_range[col]['max'] - self.num_cols_range[col]['min'])

            else:
                for col in self.num_cols_range:
                    X_df[col] = (X_df[col] - self.num_cols_range[col]['min']) / (self.num_cols_range[col]['max'] - self.num_cols_range[col]['min'])
                    X_df[col] = np.clip(X_df[col], 0, 1)

        if one_hot_cat_cols:
            # Avoid the Dummy Variable Trap
            # https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a
            
            cat_cols = X_df.select_dtypes(CAT_COLS_DTYPES).columns.to_list()
            print(cat_cols)
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

            if fit:
                self.train_cols = X_df.columns.tolist()

            else:
                test_cols = X_df.columns.tolist()
                missing_cols = set(self.train_cols) - set(test_cols)
                for col in missing_cols:
                    X_df[col] = 0

                X_df = X_df[self.train_cols]

        return X_df, y_df

NUM_COLS_DTYPES = ['int', 'float', 'datetime']
CAT_COLS_DTYPES = ['category', 'bool']


class SampleMethod(Method):
    def __init__(self, dtype, random_state=None, *args, **kwargs):
        self.dtype = dtype
        # self.smoothing = smoothing
        # self.proper = proper
        self.random_state = random_state

    def fit(self, y_df=None, *args, **kwargs):
        # if self.proper:
        #     y_df = proper(y_df=y_df)
        if self.dtype in NUM_COLS_DTYPES:
            self.x_real_min, self.x_real_max = np.min(y_df), np.max(y_df)

        self.values = y_df.to_numpy()

    def predict(self, X_test_df):
        n = X_test_df.shape[0]

        y_pred = np.random.choice(self.values, size=n, replace=True)

        # if self.smoothing and self.dtype in NUM_COLS_DTYPES:
        #     y_pred = smooth(self.dtype, y_pred, self.x_real_min, self.x_real_max)

        return y_pred

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


NUM_COLS_DTYPES = ['int', 'float', 'datetime', 'float64']
CAT_COLS_DTYPES = ['category', 'bool']

class CARTMethod(Method):
    def __init__(self, dtype, minibucket=5, random_state=None, *args, **kwargs):
        self.dtype = dtype
        self.minibucket = minibucket
        self.random_state = random_state

        if self.dtype in CAT_COLS_DTYPES:
            self.cart = DecisionTreeClassifier(min_samples_leaf=self.minibucket, random_state=self.random_state)
        if self.dtype in NUM_COLS_DTYPES:
            self.cart = DecisionTreeRegressor(min_samples_leaf=self.minibucket, random_state=self.random_state)

    def fit(self, X_df, y_df):
        
        # X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=False, one_hot_cat_cols=False)
        # if self.dtype in NUM_COLS_DTYPES:
        #     self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)
      
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.cart.fit(X, y)

        # save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)
        leaves_y_df = pd.DataFrame({'leaves': leaves, 'y': y})
        self.leaves_y_dict = leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()

    def predict(self, X_test_df):
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=False, one_hot_cat_cols=False, fit=False)

        # predict the leaves and for each leaf randomly sample from the observed values
        X_test = X_test_df.to_numpy()
        leaves_pred = self.cart.apply(X_test)
        y_pred = np.zeros(len(leaves_pred), dtype=object)

        leaves_pred_index_df = pd.DataFrame({'leaves_pred': leaves_pred, 'index': range(len(leaves_pred))})
        leaves_pred_index_dict = leaves_pred_index_df.groupby('leaves_pred').apply(lambda x: x.to_numpy()[:, -1]).to_dict()
        # print(leaves_pred_index_dict.items())
        for leaf, indices in leaves_pred_index_dict.items():
            np.random.seed(0)
            y_pred[indices] = np.random.choice(self.leaves_y_dict[leaf], size=len(indices), replace=True)

        return y_pred

import time
from sklearn.neighbors import NearestNeighbors

METHODS_MAP = {'cart': CARTMethod, 'sample': SampleMethod}

class MCCE:
    def __init__(self,
                 fixed_features=None,
                 fixed_features_encoded=None,
                 continuous=None,
                 categorical=None,
                 model=None,
                 seed=None
                 ):

        # initialise arguments
        self.fixed_features = fixed_features  # features to condition on - the ones in the dataset
        self.fixed_features_encoded = fixed_features_encoded
        self.continuous = continuous
        self.categorical = categorical

        self.seed = seed
        self.model = model

        self.method = None
        self.visit_sequence = None
        self.predictor_matrix = None
        

    def fit(self, df, dtypes=None):

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes
        self.mutable_features = [col for col in self.df_columns if (col not in self.fixed_features_encoded)]
        self.cont_feat = [feat for feat in dtypes.keys() if dtypes[feat] != 'category']


        self.n_fixed, self.n_mutable = len(self.fixed_features_encoded), len(self.mutable_features)
        
        # column indices of mutable features
        self.visit_sequence = [index for index, col in enumerate(self.df_columns) if (col in self.fixed_features_encoded)] # if (col in self.mutable_features)
        for index, col in enumerate(self.df_columns):
            if col in self.mutable_features:
                self.visit_sequence.append(index)


        # convert indices to column names
        self.visit_sequence = [self.df_columns[i] for i in self.visit_sequence]

        self.visited_columns = [col for col in self.df_columns if col in self.visit_sequence]
        self.visit_sequence = pd.Series([self.visit_sequence.index(col) for col in self.visited_columns], index=self.visited_columns)

        # create list of methods to use - currently only cart implemented
        self.method = []
        for col in self.visited_columns:
            if col in self.fixed_features_encoded:
                self.method.append('sample') # these will be fit but not sampled 
            else:
                self.method.append('cart')
        self.method = pd.Series(self.method, index=self.df_columns)

        # predictor_matrix_validator:
        self.predictor_matrix = np.zeros([len(self.visit_sequence), len(self.visit_sequence)], dtype=int)
        self.predictor_matrix = pd.DataFrame(self.predictor_matrix, index=self.visit_sequence.index, columns=self.visit_sequence.index)
        visited_columns = []
        for col, _ in self.visit_sequence.sort_values().iteritems():
            self.predictor_matrix.loc[col, visited_columns] = 1
            visited_columns.append(col)
        
        # fit
        self._fit(df)

    def _fit(self, df):
        self.saved_methods = {}
        self.trees = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, _ in self.visit_sequence.sort_values().iteritems():
            
            # initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], random_state=self.seed)
            
            # fit the method
            
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            if len(col_predictors) == 0:
              col_predictors = [col]
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            # save the method
            if self.method[col] == 'cart':
                self.trees[col] = col_method.leaves_y_dict
            self.saved_methods[col] = col_method



    def generate(self, test, k):

        self.k = k
        n_test = test.shape[0]

        # create data set with the fixed features repeated k times
        synth_df = test[self.fixed_features_encoded]
        synth_df = pd.concat([synth_df] * self.k)
        synth_df.sort_index(inplace=True)

        # repeat 0 for mutable features k times
        synth_df_mutable = pd.DataFrame(data=np.zeros([self.k * n_test, self.n_mutable]), columns=self.mutable_features, index=synth_df.index)
        synth_df = pd.concat([synth_df, synth_df_mutable], axis=1)
        # print(synth_df.head(10))
        start_time = time.time()
        for col in self.mutable_features: # self.visit_sequence.sort_values().iteritems():
            # reload the method
            col_method = self.saved_methods[col]
            # predict with the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            if len(col_predictors) == 0:
              col_predictors = [col]
            synth_df[col] = col_method.predict(synth_df[col_predictors])
            # map dtype to original dtype
            synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

        self.total_generating_seconds = time.time() - start_time

        # return in same ordering as original dataframe
        # synth_df = synth_df[test.columns]
        return synth_df

    def postprocess(self, data, synth, test, response, inverse_transform=None, cutoff=0.5):
        
        self.cutoff = cutoff

        # Predict response of generated data
        synth[response] = self.model.predict(synth)
        synth_positive = synth[synth[response]>=cutoff] # drop negative responses
        
        # Duplicate original test observations N times where N is number of positive counterfactuals
        n_counterfactuals = synth_positive.groupby(synth_positive.index).size()
        n_counterfactuals = pd.DataFrame(n_counterfactuals, columns = ['N'])

        test_repeated = test.copy()

        test_repeated = test_repeated.join(n_counterfactuals)
        test_repeated.dropna(inplace = True)

        test_repeated = test_repeated.reindex(test_repeated.index.repeat(test_repeated.N))
        test_repeated.drop(['N'], axis=1, inplace=True)

        self.test_repeated = test_repeated

        self.results = self.calculate_metrics(synth=synth_positive, test=self.test_repeated, data=data, \
            response=response, inverse_transform=inverse_transform) 

        ## Find the best row for each test obs

        results_sparse = pd.DataFrame(columns=self.results.columns)

        for idx in list(set(self.results.index)):
            idx_df = self.results.loc[idx]
            if(isinstance(idx_df, pd.DataFrame)): # If you have multiple rows
                sparse = min(idx_df.L0) # 1) find least # features changed
                sparse_df = idx_df[idx_df.L0 == sparse] 
                closest = min(sparse_df.L2) # find smallest Gower distance
                close_df = sparse_df[sparse_df.L2 == closest]

                if(close_df.shape[0]>1):
                    highest_feasibility = max(close_df.feasibility) #  3) find most feasible
                    close_df = close_df[close_df.feasibility == highest_feasibility].head(1)

            else: # if you have only one row - return that row
                close_df = idx_df.to_frame().T
                
            results_sparse = pd.concat([results_sparse, close_df], axis=0)

        self.results_sparse = results_sparse

    def calculate_metrics(self, synth, test, data, response, inverse_transform):

        features = synth.columns.to_list()
        features.remove(response)
        synth.sort_index(inplace=True)

        if inverse_transform:  # necessary for violation rate
            df_decoded_cfs = inverse_transform(synth.copy())
            df_decoded_factuals = inverse_transform(test.copy())

        else:
            df_decoded_cfs = synth.copy()
            df_decoded_factuals = test.copy()


        synth_metrics = synth.copy()
        
        # 1) Distance: Sparsity and Euclidean distance
        factual = test[features]#.sort_index().to_numpy()
        counterfactuals = synth[features]#.sort_index().to_numpy()
        
        time1 = time.time()
        distances = pd.DataFrame(metrics.distance(counterfactuals, factual, self.model), index=factual.index)

        time2 = time.time()
        self.distance_cpu_time = time2 - time1
        synth_metrics = pd.concat([synth_metrics, distances], axis=1)

        # 2) Feasibility 
        cols = data.columns.to_list()
        cols.remove(response)

        time1 = time.time()
        synth_metrics['feasibility'] = metrics.feasibility(counterfactuals, data, cols)
        
        time2 = time.time()
        self.feasibility_cpu_time = time2 - time1

        # 3) Success
        synth_metrics['success'] = 1

        # 4) Violation
        time1 = time.time()
        violations = metrics.constraint_violation(df_decoded_cfs, df_decoded_factuals, \
            self.continuous, self.categorical, self.fixed_features)
        
        synth_metrics['violation'] = violations
        time2 = time.time()
        self.violation_cpu_time = time2 - time1

        return synth_metrics

