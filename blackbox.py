import torch, os
import torch.nn as nn
from tqdm import tqdm
from utils import *
import pandas as pd
from sklearn.linear_model import LogisticRegression


class LinearClassifier(nn.Module):
  def __init__(self, clf, device):
      super(LinearClassifier, self).__init__()
      self.clf = clf 
      self.W = torch.from_numpy(self.clf.coef_).t().float()
      self.b = torch.from_numpy(self.clf.intercept_).float()
      self.device = device

  def forward(self, x):
    x = x.float()
    pred = torch.matmul(x, self.W) + self.b
    probs = torch.sigmoid(pred)
    probs = torch.cat((1-probs, probs), dim=1)
    return probs

class NeuralClassifier(nn.Module):
  def __init__(self, D, HC, device):
      super(NeuralClassifier, self).__init__()
      self.layers = nn.Sequential(
          nn.Linear(D, HC),
          nn.Dropout(0.5),
          nn.ReLU(),
          nn.Linear(HC, HC), 
          nn.Dropout(0.5),
          nn.ReLU(),
          nn.Linear(HC, 2),
          nn.Softmax(-1)
      )
      self.device = device
      
  def forward(self, x):
    return self.layers(x)
  
  def predict(self, x):
    if isinstance(x, pd.DataFrame):
      x = x.to_numpy()
    x = torch.from_numpy(x.astype('float')).float()
    x = x.to(self.device)
    probs = self.layers(x)
    out = probs.argmax(-1)
    return out.detach().cpu().numpy()

  def predict_proba(self, x):
    if isinstance(x, pd.DataFrame):
      x = x.to_numpy()
    x = torch.from_numpy(x.astype('float')).float()
    x = x.to(self.device)
    out = self.layers(x)
    return out.detach().cpu().numpy()


def load_linear_classifier(dg, cls_index):
  cls_path = f'model/{dg.name}/{dg.name.upper()}_{cls_index}.pickle'
  
  if os.path.isfile(cls_path):
    print(f'Loading model {cls_index}...')
    cls = load_pickle(cls_path)
    
  else:
    print(f'Training model {cls_index}...')
    X_train, X_val, X_test, y_train, y_val, y_test = dg.transform(return_tensor=False)
    cls = LogisticRegression(class_weight='balanced')
    cls.fit(X_train, y_train)
    print(cls.score(X_test, y_test))
    write_pickle(cls, cls_path)
  
  return cls


def load_neural_classifier(dg, cls_index, epochs, device):
  from torch.utils.data import DataLoader
  cls_path = f'model/{dg.name}/{dg.name.upper()}_{cls_index}.pt'

  if dg.discretized:
    D = dg.info['index'][-1][-1]
    for col in dg.num_cols:
      D -= dg.num_dict[col] - 1
  else:
    D = dg.info['index'][-1][-1]
  if dg.name == 'admission':
    cls = NeuralClassifier(D, 40, device)
  else: 
    cls = NeuralClassifier(D, 30, device)
  cls.to(device)

  if not os.path.isfile(cls_path): 
    X_train, X_val, X_test, y_train, y_val, y_test = dg.transform(return_tensor=True)
    train_indices = list(range(X_train.size(0)))
    val_indices = list(range(X_val.size(0)))
    train_loader = DataLoader(train_indices, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_indices, batch_size=100, shuffle=False)
    print(f'Training model {cls_index}...')
    optimizer = torch.optim.Adam(cls.parameters(), lr=1e-3)
    train_val_neural(X_train, X_val, y_train, y_val, train_loader, val_loader, 
                      cls, optimizer, cls_path, epochs)

  print(f'Loading model {cls_index} ...')
  load_model(cls, None, cls_path, device)
  cls.eval()
  return cls   


def train_val_neural(X_train, X_val, y_train, y_val, train_loader, val_loader,
                  cls, optimizer, cls_path, epochs=100):
  prev_acc = 0
  device = cls.device
  for epoch in range(epochs):
    train_loss, train_acc, val_acc = 0, 0, 0 
    for idx in tqdm(train_loader): 
      x = X_train[idx, :].to(device)
      y = y_train[idx].to(device)
      yhat = cls(x)
      label = yhat.argmax(-1)
      loss = nn.CrossEntropyLoss()(yhat, y)
      acc = (label == y).sum(0)/len(idx)
      optimizer.zero_grad()     
      loss.backward()
      
      optimizer.step()
      train_loss += loss.item()
      train_acc += acc.item()
    
    for idx in tqdm(val_loader): 
      x = X_val[idx, :].to(device)   
      y = y_val[idx].to(device)
      yhat = cls(x)
      label = yhat.argmax(-1)
      acc = (label == y).sum(0)/len(idx)
      val_acc += acc.item()
    
    train_loss = train_loss/len(train_loader)
    train_acc = train_acc/len(train_loader)
    val_acc = val_acc/len(val_loader)

    
    msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train_acc: {train_acc:.3f} // Val_acc: {val_acc:.3f}"
    print(msg)
        
    if val_acc > prev_acc:
        print("Saving model ...") 
        torch.save({'model_state_dict': cls.state_dict() ,'optimizer_state_dict': optimizer.state_dict(),}, cls_path)
        prev_acc = val_acc
      
    if val_acc >= 0.950:
      break



