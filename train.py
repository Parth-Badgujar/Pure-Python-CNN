from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import NeuralNet


dataset = torchvision.datasets.MNIST(root = './datasets/',train = True,transform = torchvision.transforms.ToTensor(),)
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 32,shuffle = True) 


NN = NeuralNet()


num_epochs = 1
for epoch in range(1,num_epochs+1):
    loss_avg = 0
    print(f'Epoch {epoch}')
    print()
    for i,(x,y) in enumerate(dataloader):
        NN.optim.zero_grad()
        y = F.one_hot(y,num_classes = 10).to(torch.float32)
        y_hat = NN.forward(x)
        loss = F.cross_entropy(y_hat,y)
        loss.backward()
        NN.optim.step()
        loss_avg += loss.item()
        if i%10 == 0:
            print(f'Epoch:[{epoch}/{num_epochs}] Batch:[{i+1}/{60000//32}]  Loss : {loss_avg/(i+1)} ')
    print(f'Average Loss : {loss_avg}')
    print()
    print()

