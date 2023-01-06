from utils import *
from utils_eval import *
from data_loader import *
from trainer import train_epoch, val_epoch

import math, os, random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch, time
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

class Perturber(nn.Module):
  def __init__(self, D, HP):
    super(Perturber, self).__init__()
    self.perturber = nn.Sequential(
          nn.Linear(D, HP), 
          nn.ReLU(),
          nn.Linear(HP, HP),
          nn.ReLU(), 
          nn.Linear(HP, HP),
          nn.ReLU(), 
          nn.Linear(HP, D)
      )
  
  def forward(self, x):
    return self.perturber(x)
    

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



      self.perturber = Perturber(D, HP)
    
    def perturb(self, col_index, x, truth_x, W, probs):
      

      start, end = self.dg.info['index'][col_index]
      p_i = probs[:, col_index : col_index+1]   

      input_i = x[:, start:end]

      logits = W[:, start:end]
      curr_states = input_i.max(dim=-1).indices
      
      
      if col_index in self.dg.causal_cols:
        logit_masks = torch.ones_like(logits, device=x.device)
        bs = truth_x.size(0)
        for batch_index in range(bs):
          curr = curr_states[batch_index].item()
          logit_masks[batch_index, :curr] = -100

        logits = logits + logit_masks

      x_tilde = self.C(logits, self.smoothing)
      num_cols_count = len(self.dg.info['range']) 
      if col_index < num_cols_count:
        ts = torch.Tensor(self.dg.info['range'][col_index]).to(x.device) 
        x_tilde = torch.matmul(x_tilde, ts).mean(dim=1).unsqueeze(-1)
        x_i = truth_x[:, col_index : col_index+1]
      
      else:
        x_i = input_i

      x_tilde = torch.mul(p_i, x_tilde) + torch.mul((1-p_i), x_i) 
      self.output[col_index] = x_tilde 


    def forward(self, x, truth_x, **kwargs): 
      # x : [B, D]

      L = kwargs['L']
      mask_threshold = kwargs['mask_threshold']
      self.smoothing = kwargs['smoothing']
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
        # P = probs = mask  


      
      W = self.perturber(x)

      # If 1: change, 0: keep
      self.output = {i: None for i in range(self.N)}
      


      for i in range(self.N):
        if self.output[i] is None:
          self.perturb(i, x, truth_x, W, probs)
          
  
    
      output = list(self.output.values())
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


def infer(X_test, truth_test, model, cls, num_samples, **kwargs):
  '''
  num_samples: no. counterfactuals needed
  '''

  kwargs['L'] = 1000

  num_samples = 100
  max_iter = 10000
  N = X_test.size(0)
  
  start = time.time()
  threshold = 1.0
  finals = []
  for i in tqdm(range(N)):
    cnt_iter = 0
    x = X_test[i:i+1, ]
    
    truth_x = truth_test[i:i+1, ]

    # Obtain predictions
    y = cls(truth_x).argmax(-1).item()
    
    counters = []

    cnt_sample = 0
    while cnt_sample < num_samples and cnt_iter < max_iter:
      _, xcf, probs, _ = model(x, truth_x, **kwargs)
      yhat = cls(xcf).argmax(-1)

      if threshold < 1.0:
        sparsity_indices = torch.where(probs.mean(-1) <= threshold)[0].tolist()
      else:
        sparsity_indices = list(range(probs.size(0)))

      
      validity_indices = torch.where(yhat!=y)[0].tolist()

      indices = set(sparsity_indices) & set(validity_indices)

      if len(indices) > 0:
        indices = list(indices)
        selected = xcf[indices,]
        counters.append(selected)
        
        cnt_sample += selected.size(0) 
      
      cnt_iter += 1
    
    end = time.time()
    if cnt_sample > 1:

      xcfs = torch.concat(counters, axis=0)
      if xcfs.shape[0] < num_samples: 
        cnt = xcfs.shape[0]
        selected = random.choices(range(cnt), k = num_samples - cnt)
        xcfs = torch.concat((xcfs, xcfs[selected, :]), dim = 0)
      else:
        xcfs = xcfs[:num_samples, ]

      ycf = cls(xcfs).argmax(-1).unsqueeze(-1)
      xcfs = torch.concat((xcfs, ycf), dim = -1)
      finals.append(xcfs.detach().cpu().numpy())

  total_time = end-start
  print(total_time)
  finals = np.concatenate(finals)
  return finals 
  

if __name__ == '__main__':
  import sys
  name = sys.argv[1]
  action = sys.argv[2]
  strategy = 'default' # equal-frequency discretization

  # Loading model and data
  dg, immutable_cols = load_data(name, True, device, strategy)

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
    output_path = f'data/{name}/output_{cls_index+1}.csv'
    if action == 'train':

      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
      scheduler = None
      

      if os.path.isfile(model_path): 
        print('Loading pre-trained model ...')
        load_model(model, None, model_path, device)

      model.to(device)
      criterion = Criterion(classifiers[cls_index], beta=1e-3) 
      epochs = 100
      train_model(model, optimizer, scheduler, model_path, criterion, epochs, kwargs)

    else: 
      print(f'Inference begins for {name} dataset ..')
      
      num_samples = 100
      load_model(model, None, model_path, device)
      output = infer(X_test, truth_test, model, num_samples, output_path, **kwargs)
      _, _, X_base, _, _, _ = dg.transform(False, dg.num_cols)
      output = pd.DataFrame(output, columns = X_base.columns.tolist() + [dg.target_col])
      output.to_csv(output_path)
    
    


  
  
