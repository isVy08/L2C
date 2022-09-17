import sys, os
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn as nn

def predict_single(arr, scaler, clf, pass_scaler=True):
    if isinstance(arr, tuple) or isinstance(arr, list):
        arr = np.array(arr)
    if pass_scaler:
        arr = scaler.transform(arr.reshape(1, -1))
    else:
        arr = arr.reshape(1, -1)
    return clf.predict(arr)[0]


def architecture(parameter, dataset, y, drop_, X, random_state, drop=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)
    if parameter == 3:      # used in dice-gradient. 
        return dataset, scaler, X_test, X_train, y_train, y_test
    # np.random.seed(42)
    # random.seed(42)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 3), max_iter=10000, random_state=random_state)
    clf.fit(X_train_, y_train)

    if parameter:
        if drop:
            dataset = dataset.drop(columns=[*drop_])
        if parameter in [1, "policy_iteration"]:
            return clf, dataset, scaler, X_test, X_train
        elif parameter == 2:    # This is used in dice-gradient.py
            return clf, dataset, scaler, X_test, X_train, y_train, y_test
        else:
            raise ValueError("Parameter must be 1 or 2")
    Y_test_pred = clf.predict(X_test_)
    Y_train_pred = clf.predict(X_train_)
    tn, fp, fn, tp = confusion_matrix(y_test, Y_test_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_train, Y_train_pred).ravel()
    print(random_state, clf.score(X_test_, y_test))
    print(random_state, (Y_test_pred == 0).sum(), (Y_test_pred == 0).sum() + (Y_train_pred == 0).sum())



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
    x = torch.from_numpy(x).float()
    x = x.to(self.device)
    probs = self.layers(x)
    out = probs.argmax(-1)
    return out.detach().cpu().numpy()

  def predict_proba(self, x):
    x = torch.from_numpy(x).float()
    x = x.to(self.device)
    out = self.layers(x)
    return out.detach().cpu().numpy()



######### LOAD MY DATA #########
def load_model_german(model_id):
    import pickle
    data_path = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/my_german_data.pickle"
    file = open(data_path, 'rb')
    X_train, X_val, X_test, encoded_columns, immutable_features = pickle.load(file)
    model_path = f"{os.path.dirname(os.path.realpath(__file__))}/models/GERMAN_{model_id}.pickle"
    file = open(model_path, 'rb')
    clf = pickle.load(file)
    return clf, X_train, X_val, X_test, encoded_columns, immutable_features

def load_model_student(model_id):
    import pickle
    data_path = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/my_student_data.pickle"
    file = open(data_path, 'rb')
    X_train, X_val, X_test, encoded_columns, immutable_features = pickle.load(file)
    model_path = f"{os.path.dirname(os.path.realpath(__file__))}/models/STUDENT_{model_id}.pickle"
    file = open(model_path, 'rb')
    clf = pickle.load(file)
    return clf, X_train, X_val, X_test, encoded_columns, immutable_features

def load_model_admission(model_id):
    import pickle
    data_path = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/my_admission_data.pickle"
    file = open(data_path, 'rb')
    X_train, X_val, X_test, encoded_columns, immutable_features = pickle.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = X_train.shape[1]    
    clf = NeuralClassifier(D, 40, device)
    model_path = f"{os.path.dirname(os.path.realpath(__file__))}/models/ADMISSION_{model_id}.pt"
    checkpoint = torch.load(model_path, map_location=device)
    clf.load_state_dict(checkpoint['model_state_dict'])
    clf.device = device
    clf.to(device)
    return clf, X_train, X_val, X_test, encoded_columns, immutable_features

def load_model_sba(model_id):
    import pickle
    data_path = f"{os.path.dirname(os.path.realpath(__file__))}/datasets/my_sba_data.pickle"
    file = open(data_path, 'rb')
    X_train, X_val, X_test, encoded_columns, immutable_features = pickle.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = X_train.shape[1]    
    clf = NeuralClassifier(D, 30, device)
    model_path = f"{os.path.dirname(os.path.realpath(__file__))}/models/SBA_{model_id}.pt"
    checkpoint = torch.load(model_path, map_location=device)
    clf.load_state_dict(checkpoint['model_state_dict'])
    clf.device = device
    clf.to(device)
    return clf, X_train, X_val, X_test, encoded_columns, immutable_features

if __name__ == "__main__":
    load_model_german(1)
    load_model_admission(1)
    load_model_student(1)
    load_model_sba(1)
