import torch
import numpy as np
import pandas as pd
import scipy

def cont_prox(num_cols, num_samples, s0, samples):
  # Lower is better
  dist = 0
  for col in num_cols:
    dist += np.abs(s0[col] - samples[col]).sum()
  return dist / (num_samples * len(num_cols)) 

def cat_prox(cat_cols, num_samples, s0, samples):
  # Lower is better
  dist = 0
  for col in cat_cols:
    dist += (s0[col] != samples[col]).sum()
  
  return dist / (num_samples * len(cat_cols)) 
  dist = 0

def sparsity(all_cols, num_samples, s0, samples):
  dist = 0
  for col in all_cols:
    dist += (s0[col] != samples[col]).sum()
  
  return dist / (num_samples * len(all_cols)) 
  dist = 0


def validity(target_col, num_samples, s0, samples):
  return (s0[target_col] != samples[target_col]).sum() / num_samples

def find_manifold_dist(x, knn):
  # x: np.array
  nearest_dist, nearest_points = knn.kneighbors(x, 1, return_distance=True)
  quantity = np.mean(nearest_dist)		
  return quantity

def diversity(dg, s0, samples, use_raw=False):
  # Higher is better
  new_samples = {}
  # Discretize numeric columns, on all datasets for evaluation
  df = dg.raw_df if use_raw else dg.df
  for col in dg.num_cols:
    column, _ = pd.qcut(df[col], dg.num_dict[col], retbins=True, duplicates='drop')
    cats = pd.Categorical(column).categories.to_list()
    new_samples[col] = np.array([dg.map_interval(x, cats) for x in samples[col]])
  

  for col in dg.cat_cols + [dg.target_col]:
    new_samples[col] = samples[col]

  cnt = 0
  div = 0
  sample_df = pd.DataFrame.from_records(new_samples)
  sample_df = sample_df.loc[sample_df[dg.target_col] != s0[dg.target_col], :] 

  # remove invalid samples 
  num_samples = sample_df.shape[0]
  for i in range(num_samples-1):
    for j in range(i+1, num_samples):
      div += scipy.spatial.distance.hamming(sample_df.iloc[i, ], sample_df.iloc[j, ] )
      cnt += 1
  
  if cnt == 0:
    return 0
  else:
    return div / cnt

def evaluate(dg, num_samples, s0, samples, use_raw):
  co = cont_prox(dg.num_cols, num_samples, s0, samples)
  ca = cat_prox(dg.cat_cols, num_samples, s0, samples)
  va = validity(dg.target_col, num_samples, s0, samples)
  di = diversity(dg, s0, samples, use_raw)
  sp = sparsity(dg.num_cols + dg.cat_cols, num_samples, s0, samples)
  return co, ca, di, va, sp

def parse_sample(dg, x):
  '''
  x : Torch tensor
  '''
  if isinstance(x, torch.Tensor):
    arr = x.cpu().detach().numpy()
  else:
    arr = x

  samples = {}
  valid_cat = 0

  _cols = dg.num_cols + dg.cat_cols

  n = len(dg.num_cols)
  for i in range(n):
    value = arr[:, i]
    col = dg.num_cols[i]
    samples[col] = value
  
  i = j = i + 1
  info = dg.info['index']
  while i < len(info):
    start, end = info[i]
    step = end - start
    col = _cols[i] 
    arr_ = arr[:, j: j+step]
    index = arr_.argmax(-1)
    samples[col] = index

    v1 = np.ones_like(arr_.sum(1))
    valid_cat += (arr_.sum(1) == v1).mean()
    j += step
    i += 1

  return samples, valid_cat / len(dg.cat_cols)

def get_clean_samples(sample_df, dg, cls, revert=False):
  
  if dg.target_col in sample_df.columns:
    arr = sample_df.astype('float').drop(columns=dg.target_col).to_numpy()
  else:
    arr = sample_df.astype('float').to_numpy()

  samples, valid_cat = parse_sample(dg, arr) 

  if revert:
    cf_df = pd.DataFrame.from_dict(samples)
    arr = []
    for col in cf_df.columns:
      if col in dg.cat_cols:
        a = dg.scaler[col][0].transform(cf_df[[col]])
      else:
        a = cf_df[[col]]
    
      arr.append(a)
    
    arr = np.concatenate(arr, axis=1)
    
  samples[dg.target_col] = cls.predict(arr)  
  
  return samples, valid_cat




