import pandas as pd
import numpy as np
import random, os, torch
from utils import load_pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class DataGenerator(object):
  def __init__(self, name, num_cols, target_col, 
                immutable_cols, discretized, device, 
                strategy = 'default', 
                causal_relations = None,
                quasi_identifiers = None
                ):
    
    self.name = name 
    self.num_cols = num_cols
    
    
    self.discretized = discretized 
    
    self.strategy = strategy if name != 'adult' else 'user'
    if self.strategy != 'default':
      # user-defined bins
      print(f'Using {self.strategy} bins ...')
      self.num_bins = load_pickle(f'data/{self.name}/{self.name}_{self.strategy}.bins') 
    else: 
      print('Using equal-density bins ...')
      self.num_bins = {'german': 4, 'admission': 3, 'student': 3, 'adult': 3}
   
 
    self.cat_cols = []
    self.target_col = target_col 
    self.immutable_cols = immutable_cols
    self.device = device

    self.scaler  = {}
    self.raw_df = None
    self.df = None
    self.train_size = None
    
    self.num_dict = {}
    self.preprocess()


    self.info = self.get_info()
    self.encoded_columns = []
    self.quasi_identifiers = quasi_identifiers
  

    # Split train test
    self.train_dataset = self.df.iloc[:self.train_size, ]
    self.test_dataset = self.df.iloc[self.train_size:, ]
    self.y_train = self.df.iloc[:self.train_size, -1]
    self.y_test = self.df.iloc[self.train_size:, -1]

    self.immutable_cols = self.update_col_names(self.immutable_cols) 
    
    cols_ = self.num_cols + self.cat_cols
    self.causal_cols = [cols_.index(col) for col in causal_relations]

            

  def update_col_names(self, col_list):
    updated = []
   
    for col in col_list:
      if (col in self.cat_cols) or (col in self.num_cols and self.discretized): 
        cats = self.df[col].cat.categories 
        for c in cats:
          updated.append(col + '_' + str(c))  
      elif col in self.num_cols and not self.discretized: 
        updated.append(col)
    
    return updated

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
      end = start + len(self.scaler[col][1])
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
  
  def map_interval(self, x, cats, return_index=True):
    '''
    cats : a list of pandas Interval objects in increasing order
    '''
    if x <= cats[0].left:
      return 0 if return_index else cats[0]
    
    if x > cats[-1].right:
      return len(cats) - 1 if return_index else cats[-1]
    
    for i, interval in enumerate(cats):
      if x > interval.left and x <= interval.right:
        return i if return_index else interval

  def preprocess(self):
    '''
    Discretize continuous columns and re-index categorical columns
    '''

    train_path = f'data/{self.name}/{self.name}_train.csv'
    test_path = f'data/{self.name}/{self.name}_test.csv'

    if self.name == 'admission' and not os.path.isfile(train_path):
      print('Train-test split for Admission data ...')
      df = pd.read_csv('data/admission/admission.csv')
      df.drop(columns=['Serial No.'], inplace = True)
      df['Chance of Admit '] = df['Chance of Admit '].map(lambda x: 1 if x >= 0.7 else 0).to_numpy()
      df.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)
      train_size = int(df.shape[0] * 0.8) 
      df.iloc[:train_size, :].to_csv(train_path, index=False)
      df.iloc[train_size:, :].to_csv(test_path, index=False)  
    
    
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
      
   
    if self.name in ('sba','adult'):
      if self.name == 'sba':
          drop_col = 'Selected'
          train_dataset[self.target_col] = train_dataset[self.target_col].astype('int')
          test_dataset[self.target_col] = test_dataset[self.target_col].astype('int') 
      elif self.name == 'adult':
          drop_col = 'fnlwgt'
      
      train_dataset.drop(columns=[drop_col], inplace=True)
      test_dataset.drop(columns=[drop_col], inplace=True)
       
      

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
      df[col] = mapper.fit_transform(df[[col]])
      train_dataset[col] = mapper.transform(train_dataset[[col]])
      num_to_cats = None
      self.raw_df[col] = df[col]

      if self.discretized:
        if self.strategy == 'default':
          # binning by quartiles
          column = pd.qcut(df[col], self.num_bins[self.name], retbins=False, duplicates='drop')  
          cats = pd.Categorical(column).categories.to_list()      
        else:
          cats = [pd.Interval(left=cat[0], right=cat[1], closed='right') for cat in self.num_bins[col]]
          column = df[col].map(lambda x: self.map_interval(x, cats, return_index=False))
        
        self.num_dict[col] = len(cats)
        num_to_cats = dict(zip(cats, range(len(cats))))
        df[col] = column.map(num_to_cats)
        ncats = list(range(len(cats)))
        mapper = OneHotEncoder(handle_unknown='ignore',sparse=False, categories = [ncats] )
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
      assert threshold is not None, 'Threshold must be specified.'
      for _ in range(threshold):
        index = [random.choice(range(n)) for _ in range(size)]
        MASK[range(size), index] = 0
    else:
      MASK[:, locations] = 0
    
    return MASK
  