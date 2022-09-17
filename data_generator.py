import random, os
import torch 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class DataGenerator(object):
  def __init__(self, name, num_dict, target_col, 
                immutable_cols, discretized, device):

    self.name = name 
    self.num_cols = list(num_dict.keys())
    self.num_dict = num_dict
    self.cat_cols = []
    self.target_col = target_col 
    self.immutable_cols = immutable_cols
    self.device = device

    self.scaler  = {}
    self.raw_df = None
    self.df = None
    self.train_size = None
    self.discretized = discretized 
    self.preprocess()
    self.info = self.get_info()
    self.encoded_columns = []
  

    # Split train test
    self.train_dataset = self.df.iloc[:self.train_size, ]
    self.test_dataset = self.df.iloc[self.train_size:, ]
    self.y_train = self.df.iloc[:self.train_size, -1]
    self.y_test = self.df.iloc[self.train_size:, -1]

    self.update_immutable_cols() # how to handle immutables?

  def update_immutable_cols(self):
    updated = []
    
    if self.discretized:
      _cols = self.num_cols + self.cat_cols 
    else:
      _cols = self.cat_cols
    
    for col in self.immutable_cols:
      if col in _cols:
        cats = self.df[col].cat.categories
        for c in cats:
          updated.append(col + '_' + str(c))
    
    self.immutable_cols = updated

  def get_info(self):
    info = {'index': [], 'range': []}
    if self.discretized:
      _cols = self.num_cols + self.cat_cols 
      start = 0
    else:
      _cols = self.cat_cols
      start = len(self.num_cols)
      info['index'].extend([1 for i in range(start)])
 
    
    for col in _cols:
      end = start + self.df[col].unique().shape[0]
      info['index'].append((start, end))
      start = end
      if col in self.num_cols:
        tensor = [[interval.left, interval.right] for interval in self.scaler[col][1]]
        info['range'].append(tensor)

    return info
  
  def transform(self, return_tensor=False, excluded_cols=[]):
    '''
    excluded_cols: features not subject to one hot encoding
                  if continuous columns, revert to standardized values
    '''
    self.encoded_columns = []
    print('Transforming data ...')
    val_size = int(0.2 * self.train_size)
    train_size = self.train_size - val_size
    
    X = []

    for col in self.num_cols + self.cat_cols:
      mapper = self.scaler[col][0]
      if self.scaler[col][1] is not None and col not in excluded_cols:
        encoded = mapper.transform(self.df[[col]])
      elif col in self.num_cols:
        encoded = self.raw_df[col].to_numpy().reshape(-1, 1)
      else:
        encoded = self.df[col].to_numpy().reshape(-1, 1)

      X.append(encoded)
      enc_cols = mapper.get_feature_names_out().tolist()
      self.encoded_columns.extend(enc_cols)
  
    X = np.concatenate(X, axis=1)


    x_train = X[:train_size, :]
    x_val = X[train_size:self.train_size, ]
    x_test = X[self.train_size:, ]

    y_train =  self.y_train.iloc[:train_size,]
    y_val =  self.y_train.iloc[train_size:self.train_size, ]
    y_test = self.y_test

    if return_tensor: 
      x_train = torch.from_numpy(x_train).float().to(self.device)
      x_val = torch.from_numpy(x_val).float().to(self.device)
      x_test = torch.from_numpy(x_test).float().to(self.device)
      y_train = torch.from_numpy(y_train.to_numpy()).to(self.device)
      y_val = torch.from_numpy(y_val.to_numpy()).to(self.device)
      y_test = torch.from_numpy(y_test.to_numpy()).to(self.device)
    
    else:
      x_train = pd.DataFrame(x_train, columns=self.encoded_columns)
      x_val = pd.DataFrame(x_val, columns=self.encoded_columns)
      x_test = pd.DataFrame(x_test, columns=self.encoded_columns)
  


    return x_train, x_val, x_test, y_train, y_val, y_test
  
  def map_interval(self, x, cats):
    '''
    cats : a list of pandas Interval objects in increasing order
    '''
    if x < cats[0].left:
      return 0
    
    if x > cats[-1].right:
      return len(cats) - 1
    
    for i, interval in enumerate(cats):
      if x > interval.left and x <= interval.right:
        return i

  def preprocess(self):
    '''
    Discretize continuous columns and re-index categorical columns
    '''

    train_path = f'data/{self.name}_train.csv'
    test_path = f'data/{self.name}_test.csv'

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    if self.name == 'sba':
      train_dataset.drop(columns=['Selected'], inplace=True)
      test_dataset.drop(columns=['Selected'], inplace=True)
      train_dataset[self.target_col] = train_dataset[self.target_col].astype('int')
      test_dataset[self.target_col] = test_dataset[self.target_col].astype('int') 
    
    self.train_size = train_dataset.shape[0]
    print('Train size: ', train_dataset.shape, 'Test size: ', test_dataset.shape)
    df = pd.concat((train_dataset, test_dataset))
    df.reset_index(inplace=True, drop=True)

    # Data before one-hot encoding
    self.raw_df = df.copy()

    # Obtain categorical features
    for col in df.columns:
      if col not in self.num_cols + [self.target_col]:
        self.cat_cols.append(col)

    # Preprocess data
    for col in self.num_cols:

      # Always standardize first, 
      mapper = StandardScaler()
      train_dataset[col] = mapper.fit_transform(train_dataset[[col]])
      df[col] = mapper.transform(df[[col]])
      num_to_cats = None
      self.raw_df[col] = df[col]

      if self.discretized:
        column = pd.qcut(df[col], self.num_dict[col], retbins=False, duplicates='drop')
        
        # Exceptional treatment for Portion feature of SBA dataset to obtain larger buckets 
        if (col == 'Portion' and self.name == 'sba'):
          column = pd.cut(df[col], self.num_dict[col], retbins=False, duplicates='drop')
        
        cats = pd.Categorical(column).categories.to_list()
        if len(cats) != self.num_dict[col]:
          self.num_dict[col] = len(cats)

        num_to_cats = dict(zip(cats, range(len(cats))))
        df[col] = df[col].map(lambda x: self.map_interval(x, cats)) 

        mapper = OneHotEncoder(handle_unknown='ignore',sparse=False)
        mapper.fit(df[[col]])
        df[col] = df[col].astype('category')
      self.scaler[col] = (mapper, num_to_cats)
  
    for col in self.cat_cols:
      cats = pd.Categorical(df[col]).categories.to_list()
      num_to_cats = dict(zip(cats, range(len(cats))))
      df[col] = df[col].map(num_to_cats)
      mapper = OneHotEncoder(handle_unknown='ignore', sparse=False)
      mapper.fit(df[[col]])
      df[col] = df[col].astype('category')
      self.scaler[col] = (mapper, num_to_cats)

    # Reorder columns
    df = df[self.num_cols + self.cat_cols + [self.target_col]]
    self.df = df
    self.raw_df = self.raw_df[self.num_cols + self.cat_cols + [self.target_col]]
    
  

  def create_mask(self, size, threshold=None, locations=None):
    n = len(self.info['index'])
    MASK = torch.ones((size, n), device=self.device)
    if locations is None:
      '''
      This is used to create random masks with a certain threshold specifying max no. features to be kept fixed.
      '''
      assert threshold is not None, 'Threshold must be specified.'
      for _ in range(threshold):
        index = [random.choice(range(n)) for _ in range(size)]
        MASK[range(size), index] = 0
    else:
      MASK[:, locations] = 0
    
    return MASK
  