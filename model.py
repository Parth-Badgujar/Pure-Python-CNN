from layers import Conv2d, Maxpool2d, Linear, Flatten
from activations import relu
import torch 

class NeuralNet():
    def __init__(self):
        self.conv1 = Conv2d(1,16,(3,3))
        self.conv2 = Conv2d(16,32,(3,3))
        self.conv3 = Conv2d(32,64,(3,3))
        self.mp2d = Maxpool2d((2,2),(2,2))
        self.fc1 = Linear(576,64)
        self.fc2 = Linear(64,10)
        self.optim = torch.optim.Adam([self.conv1.weights,
                                     self.conv2.weights,
                                     self.conv3.weights,
                                     self.fc1.weights,
                                     self.fc2.weights])
    def forward(self,x):
        x = relu(self.conv1.forward(x))
        x = self.mp2d.forward(x)
        x = relu(self.conv2.forward(x))
        x = self.mp2d.forward(x)
        x = relu(self.conv3.forward(x))
        x = Flatten(x)
        x = relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x