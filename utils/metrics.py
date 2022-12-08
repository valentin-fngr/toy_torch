import torch  
import torch.nn as nn 


def accuracy(logits, target):

    preds = nn.functional.softmax(logits) 
    preds = torch.argmax(preds, axis=1) 
    acc = torch.where(preds == target, 1, 0).to(torch.float32).mean()
    return acc
    