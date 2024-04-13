import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):   
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
          
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)     
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class MultiLoss(nn.Module):
    def __init__(self, loss_weights=None):
        super(MultiLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, *losses):
        total_loss = 0.0
        if self.loss_weights is None:
            self.loss_weights = [1.0] * len(losses)
        for i, loss in enumerate(losses):
            total_loss += self.loss_weights[i] * loss
        return total_loss
    
class DynamicWeightAverage:
    def __init__(self, n_losses, alpha=2.0):
        self.weights = [1.0 / n_losses] * n_losses
        self.alpha = alpha  

    def update(self, loss_gradients):
        normed_grads = [torch.norm(g.detach(), 2).pow(self.alpha).item() for g in loss_gradients]
        normed_sum = sum(normed_grads)

        new_weights = [ng / normed_sum for ng in normed_grads]

        self.weights = new_weights