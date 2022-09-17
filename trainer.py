from tqdm import tqdm

def train_epoch(model, optimizer, scheduler, loader, X, truth, criterion, device, kwargs):
    model.train()
    losses = 0
    accs = 0
    L = kwargs['L']
    for idx in tqdm(loader): 
        x = X[idx, :].to(device) 
        truth_x = truth[idx, :].to(device) 
        preds = model(x, truth_x, **kwargs)
        
        loss, acc = criterion(preds)

        optimizer.zero_grad()       
        loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), 0.01)
        optimizer.step()
        if scheduler is not None:
          scheduler.step()
        


        losses += loss.item()
        accs += acc.item()

    return losses / len(loader), accs / len(loader)

def val_epoch(model, loader,  X, truth, criterion, device, kwargs):
    model.eval()
    losses = 0
    accs = 0
    for idx in tqdm(loader): 

        x = X[idx, :].to(device)  
        truth_x = truth[idx, :].to(device)   
        preds = model(x, truth_x, **kwargs)
        loss, acc = criterion(preds)

        losses += loss.item()
        accs += acc.item()

    return losses / len(loader), acc / len(loader)