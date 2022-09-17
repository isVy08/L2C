import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, cls, alpha=1e-4):
        super(Criterion, self).__init__()
        self.cls = cls
        self.alpha = alpha
    
    def forward(self, preds):

        for param in self.cls.parameters():
            param.requires_grad = False

        
        x, xcf, probs, logits = preds
        target_label = self.cls(x).argmin(-1)  
        
        n = xcf.size(0)
        ycf = self.cls(xcf)
        
        loss = self.validity_loss(ycf, target_label, 'ce')
        sparsity = torch.norm(probs, p=1) 
        loss += self.alpha * sparsity

        y = ycf.argmax(-1)
        acc = (y == target_label).sum()/len(y)
        acc = acc.mean()
        return loss, acc
    

    def validity_loss(self, y, target_label, loss):
        if loss == 'ce': 
            ce = nn.CrossEntropyLoss()
            return ce(y, target_label)
        elif loss == 'hinge':
            level = 0.4
            target = torch.zeros_like(target_label) + level
            indices = target_label.nonzero()
            target[indices] = 1-level
            target = torch.stack((1-target, target), dim=1)
            return nn.CrossEntropyLoss()(y, target)
        else:
            raise ValueError


