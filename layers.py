import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride = (1, 1)):
        self.inch = in_channels
        self.outch = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weights = torch.rand((self.outch, self.inch, self.kernel_size[0], self.kernel_size[1]), requires_grad = True)
        self.bias = torch.rand(self.outch, requires_grad = True)
    def convolution(channel, filter, stride):
        filter_size = filter.shape
        if (channel.shape[0] - filter_size[0] + 1)%stride[0] == 0 and (channel.shape[1] - filter_size[1] + 1)%stride[1] == 0:
            result = torch.zeros((channel.shape[0] - filter_size[0] + 1)//stride[0], (channel.shape[1] - filter_size[1] + 1)//stride[1])
            for row in range(channel.shape[0] - filter_size[0] + 1, stride[0]):
                for coloumn in range(channel.shape[1] - filter_size[1] + 1, stride[1]):
                    matrix = channel[row:row + filter_size[0], coloumn:coloumn + filter_size[1]]
                    dot = 0
                    for i in range(len(matrix)):
                        for j in range(i):
                            dot += matrix[i][j]*filter[i][j]
                    result[i][j] = dot
            return result
        else:
            print('Convolution not possible !')
    def forward(self, batch):
        self.final_size_row = (batch.shape[2] - self.kernel_size[0] + 1)//self.stride[0]
        self.final_size_col = (batch.shape[3] - self.kernel_size[1] + 1)//self.stride[1]
        self.batch_size = len(batch)
        result = torch.zeros(self.batch_size, self.outch, self.final_size_row, self.final_size_col)
        for im in range(self.batch_size):
            for i in range(self.outch):
                img = torch.zeros(self.final_size_row, self.final_size_col)
                for j in range(self.inch):
                    img += Conv2d.convolution(batch[im][j], self.weights[i][j], self.stride)
                result[im][i] = img + self.bias[i]
        return result

class Maxpool2d():
    def __init__(self, kernel, stride):
        self.kernel_size = kernel
        self.stride = stride
    def forward(self, batch):
        self.final_size_row = (len(batch[0][0]) - self.kernel_size[0])//self.stride[0] + 1
        self.final_size_col = (len(batch[0][0][0]) - self.kernel_size[1])//self.stride[1] + 1
        self.batch_size = batch.shape[0]
        self.channels = batch.shape[1]
        self.result = torch.zeros((self.batch_size, self.channels, self.final_size_row, self.final_size_col))
        for img in range(self.batch_size):
            for channel in range(self.channels):
                for row in range(0, self.final_size_row, self.stride[0]):
                    for coloumn in range(0, self.final_size_col, self.stride[1]):
                        self.result[img][channel][row][coloumn] = torch.max(batch[img][channel][row:row + self.kernel_size[0], coloumn:coloumn + self.kernel_size[1]])
        return self.result

class Linear():
    def __init__(self, inf, outf):
        self.weights = torch.rand(inf, outf).uniform_( - 1/(inf)**0.5, 1/(inf)**0.5).requires_grad_(True)
        self.bias = torch.empty(outf).uniform_( - 1/(inf)**0.5, 1/(inf)**0.5).requires_grad_(True)
    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias  
    

