from utils import *
from utils_eval import *
from blackbox import *
from data_loader import *
from trainer import train_epoch, val_epoch

import pandas as pd
import random, math, os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import scipy, torch, time
from torch.utils.data import DataLoader
from criterion import Criterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(8)
np.random.seed(18)

class Sample_Categorical(nn.Module):

  def __init__(self, tau):
    super(Sample_Categorical, self).__init__()
    self.tau = tau
  
  def forward(self, logits, smoothing=None):
    # logits : [B, K, 1], K categories
    logits = logits.squeeze(-1)
    c = logits.size(-1)
    
    if self.training:
      if smoothing:
        pr = torch.softmax(logits, -1)
        noise = torch.ones_like(pr) / c
        pr = torch.mul(1-smoothing, pr) + torch.mul(smoothing, noise)
        logits = torch.log(pr)

      sample = nn.functional.gumbel_softmax(logits, hard=False, tau=self.tau, dim=-1)
    else:
      choice = torch.distributions.categorical.Categorical(logits=logits)
      s = choice.sample()
      sample = torch.eye(c)[s]
    return sample

class Sample_Bernoulli(nn.Module):
  def __init__(self, tau):
    super(Sample_Bernoulli, self).__init__()
    self.tau = tau
  def forward(self, probs):
    if self.training:
      unif_a = torch.rand_like(probs)
      unif_b = torch.rand_like(probs)

      gumbel_a = -torch.log(-torch.log(unif_a))
      gumbel_b = -torch.log(-torch.log(unif_b))
      no_logits = (probs * torch.exp(gumbel_a))/self.tau
      de_logits = no_logits + ((1.0 - probs) * torch.exp(gumbel_b))/self.tau
      sample = no_logits / de_logits
    else:
      sample = torch.bernoulli(probs)
    return sample

class Model(nn.Module):
    
    def __init__(self, D, HP, HS, dg, tau=0.7):
      super(Model, self).__init__()
      
      self.dg = dg
      self.N = len(dg.info['index'])
      self.C = Sample_Categorical(tau)
      self.B = Sample_Bernoulli(tau)
      
      self.selector = nn.Sequential(
          nn.Linear(D, HS), 
          nn.ReLU(),
          nn.Linear(HS, self.N),
          nn.Sigmoid()          
      )

      self.perturber = nn.Sequential(
          nn.Linear(D, HP), 
          nn.ReLU(),
          nn.Linear(HP, HP),
          nn.ReLU(), 
          nn.Linear(HP, D)
      )


    def forward(self, x, truth_x, **kwargs): 
      # x : [B, N]
      
      L = kwargs['L']
      mask_threshold = kwargs['mask_threshold']
      smoothing = kwargs['smoothing']
      mask_locations = kwargs['mask_locations']

      if L > 1: 
        x = torch.repeat_interleave(x, L, dim=0)
        truth_x = torch.repeat_interleave(truth_x, L, dim=0)
    
      P = self.selector(x)
      probs = self.B(P)
      
      if bool(mask_locations) or bool(mask_threshold):
        
        batch_size = x.size(0)
        mask = self.dg.create_mask(batch_size, mask_threshold, mask_locations)
        probs = torch.mul(probs, mask)  
      
      x_ = torch.exp(x)
      W = self.perturber(x_)

      # If 1: change, 0: keep
      output = []
      num_cols_count = len(self.dg.info['range'])
      for i in range(self.N):
        
        start, end = self.dg.info['index'][i]
        p_i = probs[:, i:i+1]        
        
        logits = W[:, start:end]
        x_tilde = self.C(logits, smoothing)
        
        # Decode to continuous rep.
        if i < num_cols_count:
          ts = torch.Tensor(self.dg.info['range'][i]).to(x.device) 
          x_tilde = torch.matmul(x_tilde, ts).mean(dim=1).unsqueeze(-1)
          
          x_i = truth_x[:, i:i+1]    
          
        else:
          x_i = x[:, start:end]

        x_tilde = torch.mul(p_i, x_tilde) + torch.mul((1-p_i), x_i)       
        output.append(x_tilde)
      
      xcf = torch.cat(output, dim=1)

      if self.training:
        return truth_x, xcf, P, W
      else:
        return truth_x, xcf, probs, W


def train_model(model, optimizer, scheduler, model_path, criterion, epochs, kwargs):
  
  start = time.time()
  prev_acc = 0
  for epoch in range(1, epochs + 1):
      train_loss, train_acc = train_epoch(model, optimizer, scheduler, train_loader, X_train, truth_train, 
                                          criterion, device, kwargs)        
      
      # Evaluation
      val_loss, val_acc = val_epoch(model, val_loader, X_val, truth_val,
                                    criterion, device, kwargs)
      

      msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train_acc: {train_acc:.3f} // Val loss: {val_loss:.3f}, Val_acc: {val_acc:.3f}"
      print(msg)

      if math.isnan(train_loss) or math.isnan(val_loss):
          break
          
      if val_acc > prev_acc:
          print("Saving model ...") 
          torch.save({'model_state_dict': model.state_dict() ,'optimizer_state_dict': optimizer.state_dict(),}, model_path)
          prev_acc = val_acc

  end = time.time()
  training_time = end-start
  print('TOTAL TRAINING TIME: ', training_time)


def infer(X_test, truth_test, model, num_samples, **kwargs):
  '''
  num_samples: no. counterfactuals needed
  '''

  kwargs['L'] = 100

  # Empirical minimum sparsity thresholds for each dataset
  TH = {'german': 0.50, 'admission': 0.70, 'student': 0.30, 'sba': 0.30}


  num_samples = 100
  max_iter = 100000
  N = X_test.size(0)
  
  CO, CA, DI, VA, SP, CG, MAN, VAC = 0, 0, 0, 0, 0, 0, 0, 0
  start = time.time()
  print(cls_index + 1)
  cls = classifiers[cls_index]

  threshold = TH[name]
  for i in tqdm(range(N)):
    cnt_iter = 0
    x = X_test[i:i+1, ]
    
    truth_x = truth_test[i:i+1, ]
    s0, _ = parse_sample(dg, truth_x)

    # Obtain predictions
    y = cls(truth_x).argmax(-1).item()
    s0[dg.target_col] = y
    
    counters = []

    cnt_sample = 0
    while cnt_sample < num_samples and cnt_iter < max_iter:
      out = model(x, truth_x, **kwargs)
      xcf = out[1]
      probs = out[2]
      yhat = cls(xcf).argmax(-1)

      
      condA = torch.where(yhat!=y)[0].tolist()

      if threshold < 1.0:
        p_mean = probs[:, :len(dg.num_cols)].mean(1)
        condB = torch.where(p_mean < threshold)[0].tolist()
      else:
        condB = condA
      
      indices = set(condA) & set(condB)
      
      if len(indices) > 0:
        indices = list(indices)
        selected = xcf[indices,]
        counters.append(selected)
        
        cnt_sample += selected.size(0) 
      
      cnt_iter += 1
    
    end = time.time()
    if cnt_sample > 1:

      xcfs = torch.concat(counters, axis=0)
      xcfs = xcfs[:num_samples, ]
      samples, vac = parse_sample(dg, xcfs)
      samples[dg.target_col] = cls(xcfs).argmax(-1).cpu().numpy()
      co, ca, di, va, sp = evaluate(dg, num_samples, s0, samples, True)
      if va > 0:
          CG += 1
          
      CO += co 
      CA += ca 
      DI += di 
      VA += va
      SP += sp
      VAC += vac

    
      output_ = xcfs[:, :len(dg.num_cols)].detach().numpy()
      MAN += find_manifold_dist(output_, knn)
  
  total_time = end - start
  return f'Cont Prox: {CO/N}, 
          Cat Prox: {CA/N}, 
          Diversity: {DI/N}, 
          Sparsity: {SP/N}, 
          Validity: {VA/N}, 
          Coverage: {CG/N}, 
          Manifold Dist: {MAN/N}, 
          Valid Cat: {VAC/N}, 
          Inference time: {total_time}'

def train_load_knn(name):
  

  if not os.path.isfile(f'model/{name}.knn'):
    print(f'Training knn for {name} dataset ...')
    dg, immutable_cols = load_data(name, False, device)
    X_train, X_val, X_test, y_train, y_val, y_test = dg.transform(return_tensor=False)
    
    train_data = np.concatenate((X_train, X_val))
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=5, p=1)
    n = len(dg.num_cols)
    train_data = train_data[:, :n]
    knn.fit(train_data)
    print(f'Writing knn for {name} dataset ...')
    write_pickle(knn, f'model/{name}.knn')
  
  else: 
    knn = load_pickle(f'model/{name}.knn')
  
  return knn


if __name__ == '__main__':
  import sys
  name = sys.argv[1]
  action = sys.argv[2]

  # Loading model and data
  dg, immutable_cols = load_data(name, discretized=True, device)

  classifiers = load_blackbox(name, dg, wrap_linear=True)

  cols_ = dg.num_cols + dg.cat_cols
  mask_locations = [cols_.index(col) for col in immutable_cols]
  print(mask_locations)


  # One hot encoded features
  X_train, X_val, X_test, y_train, y_val, y_test = dg.transform(return_tensor=True)
  print(X_train.shape, X_val.shape, X_test.shape)

  # One hot encoded categorical features only
  truth_train, truth_val, truth_test, _, _, _ = dg.transform(True, dg.num_cols)

  train_indices = list(range(X_train.size(0)))
  val_indices = list(range(X_val.size(0)))
  batch_size = 200
  train_loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_indices, batch_size=batch_size, shuffle=False)

  D = X_train.shape[1]

  

  # L is number of local samples generated per iteration to optimize networks. L = 5 in our experiments. 
  kwargs = {'mask_threshold': None, 'mask_locations': mask_locations,'L': 5, 'smoothing': 1.0}

  for cls_index in range(5):
    model = Model(D, 50, 50, dg, 0.2)
    
    if not os.path.isdir(f'model/{name}/'):
      os.makedirs(f'model/{name}')

    model_path = f'model/{name}/model_{cls_index+1}.pt'
    if action == 'train':

      

      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
      scheduler = None
      

      if os.path.isfile(model_path): 
        print('Loading pre-trained model ...')
        load_model(model, None, model_path, device)

      model.to(device)
      criterion = Criterion(classifiers[cls_index], beta=1e-4) 
      epochs = 100
      train_model(model, optimizer, scheduler, model_path, criterion, epochs, kwargs)

    else: 
      print(f'Inference begins for {name} dataset ..')
      if not os.path.isfile(model_path): 
        print('No pre-trained model exists! Specify "train" to train the model.')
      num_samples = 100
      load_model(model, None, model_path, device)
      infer(X_test, truth_test, model, num_samples, **kwargs)
    
    


  
  
