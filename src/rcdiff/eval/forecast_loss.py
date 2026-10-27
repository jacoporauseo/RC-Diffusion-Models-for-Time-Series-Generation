import torch 


def qlike(actual, pred):
    r = actual/pred
    return torch.mean(r-torch.log(r)-1)

def mse(actual, pred):
    return torch.mean((actual-pred)**2)

def mae(actual, pred):
    return torch.mean(torch.abs(actual-pred))








