import torch.nn as nn
from collections import OrderedDict


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class C5(nn.Module):
    def __init__(self):
        super(C5, self).__init__()

        self.c5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c5(img)
        return output


class F6(nn.Module):
    def __init__(self):
        super(F6, self).__init__()
        
        self.f6 = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))
    
    def forward(self, img):
        output = self.f6(img)
        return output
    
        
class FCoutput(nn.Module):
    def __init__(self):
        super(FCoutput, self).__init__()

        self.fcoutput = nn.Sequential(OrderedDict([
            ('fcoutput7', nn.Linear(84, 10)),
            ('sig1', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.fcoutput(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.c1 = C1()
        self.c3 = C3()
        self.c5 = C5()
        self.f6 = F6()
        self.fcoutput = FCoutput()
        
    def forward(self, img):
        
        # Conv Layer(C1)
        # - input:        32x32x1
        # - output:       28x28x6
        # - weights:      (5x5x1 + 1)x6
        # Sub-sampling(S2)
        # - input:        28x28x6
        # - output:       14x14x6
        # - weights:      2x2x1
        output = self.c1(img)
        
        # Conv Layer(C3)
        # - input:        14x14x6
        # - output:       10x10x16
        # - weights:      (5x5x6 + 1)x16
        # Sub-sampling(S4)
        # - input:        10x10x16
        # - output:       5x5x16
        # - weights:      2x2x1
        output = self.c3(output)
        
        # Conv Layer(C5)
        # - input:        5x5x16
        # - output:       1x1x120
        # - weights:      (5x5x16 + 1)x120
        output = self.c5(output)
        
        # Flatten Layer
        output = output.view(img.size(0), -1)
        
        # Fully Connected Layer(F6)
        # - input:        120
        # - output:       84
        output = self.f6(output)
        
        # Fully Connected Layer(F7)
        # - input:        84
        # - output:       10
        output = self.fcoutput(output)
        return output







