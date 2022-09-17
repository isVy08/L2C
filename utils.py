import os
import pickle
import torch



def load_pickle(datadir):
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

def load_model(model, optimizer, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
