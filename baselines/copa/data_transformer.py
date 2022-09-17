from dice_ml import Data
import numpy as np
import pandas as pd


class DataTransformer():
    def __init__(self, data_interface):
        self.data_interface = data_interface
        self.data_interface.create_ohe_params()

    def transform(self, x, intercept_feature=False, tonumpy=False):
        ex = self.data_interface.get_ohe_min_max_normalized_data(x)
        if intercept_feature:
            ex.insert(0, 'intercept_feature', 1.0)
        if tonumpy:
            return ex.to_numpy()
        else:
            return ex

    def inverse_transform(self, x, intercept_feature=False, tonumpy=False):
        if intercept_feature:
            if isinstance(x, pd.DataFrame):
                x = x.drop('intercept_feature', axis=1)
            else:
                x = x[:, 1:]
        ex = self.data_interface.get_inverse_ohe_min_max_normalized_data(x)

        if tonumpy:
            return ex.to_numpy()
        else:
            return ex


