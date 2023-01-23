import torch.nn as nn
import torch

# from data_loader import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# strategy = 'default'
# dg, immutable_cols = load_data(name, True, device, strategy)
# binary_relations = ['age', 'Present-residence-since', 0]
# cols_ = dg.num_cols + dg.cat_cols
# dg.binary_relations = [cols_.index(col) for col in binary_relations[:-1]] + [binary_relations[-1]]


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
      # sample = torch.softmax(logits, dim=-1)
    else:
      choice = torch.distributions.categorical.Categorical(logits=logits)
      s = choice.sample()
      s = s.to('cpu')
      sample = torch.eye(c)[s]
      sample = sample.to(logits.device)
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
      probs = torch.clamp(probs, min=0.0, max=1.0)
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
    
    def search_non_increasing_samples(self, parent_index, x):

      '''
      - if parent = 0, child = 0
      - if parent > 0, child > 0 (type 0)
      - if parent > 0, child >= 0 (type 1) 
      '''

      num_cols_count = len(self.dg.info['range'])
      x_tilde = self.output[parent_index]

      start, end = self.dg.info['index'][parent_index]
      x_i = x[:, start:end]

      before = x_i.argmax(dim=-1)
      after = x_tilde.argmax(dim=-1)
      constant_samples = torch.argwhere((after - before) == 0).squeeze(-1)  
      increasing_samples = torch.argwhere((after - before) > 0).squeeze(-1)  
      return constant_samples.tolist(), increasing_samples.tolist()
    
    
    def perturb(self, col_index, x, truth_x, W, probs):
      

      start, end = self.dg.info['index'][col_index]
      p_i = probs[:, col_index : col_index+1]   

      

      input_i = x[:, start:end]

      logits = W[:, start:end]

      curr_states = input_i.max(dim=-1).indices

      if col_index in self.dg.causal_cols or col_index == self.dg.binary_relations[1]:
        relation_type = -1
        
        if col_index == self.dg.binary_relations[1]:
          relation_type = self.dg.binary_relations[2]
          parent_index = self.dg.binary_relations[0]
          constant_samples, samples = self.search_non_increasing_samples(parent_index, x)                                            

        else:
          constant_samples = []
          bs = truth_x.size(0)
          samples = range(bs)

        
        # force unchange
        p_i_masks = torch.ones_like(p_i, device=x.device)
        p_i_masks[constant_samples, ] = 0.0
        p_i = torch.mul(p_i, p_i_masks)


        if relation_type == 0:
          # force increase: if orig == 0
          p_i_masks = torch.zeros_like(p_i, device=x.device)
          p_i_masks[samples, ] = 1.0
          p_i = p_i + p_i_masks
          # some values = 2 
          p_i = torch.clamp(p_i, 0.0, 1.0)
        
        logit_masks = torch.ones_like(logits, device=x.device)
        for batch_index in samples:
          curr = curr_states[batch_index].item()
          if relation_type == 0:
            logit_masks[batch_index, :curr+1] = -100
          else:  
            logit_masks[batch_index, :curr] = -100
        
      
        logits = logits + logit_masks
    

      x_tilde = self.C(logits, self.smoothing)
      x_tilde = torch.mul(p_i, x_tilde) + torch.mul((1-p_i), input_i) 
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
        self.perturb(i, x, truth_x, W, probs)
          
  
    
      output = list(self.output.values())
      xcf = torch.cat(output, dim=1)
      

      if self.training:
        return truth_x, xcf, P, W
      else:
        return truth_x, xcf, probs, W

