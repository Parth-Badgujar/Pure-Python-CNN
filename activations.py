import torch 

def relu(x):
    return torch.maximum(x,torch.tensor(0))
 
def sigmoid(x):
    return 1/(1+torch.exp(-x))
 
def softmax(x):
    s=torch.sum(torch.exp(x),dim=1)
    for i in range(len(x)):
        x[i]=torch.exp(x[i])/s[i]
    return x
def Flatten(x):
    x=x.view(x.shape[0],-1)
    return x