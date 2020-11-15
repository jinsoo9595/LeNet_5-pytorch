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
        self.c3_1 = C3()
        self.c3_2 = C3()
        self.c5 = C5()
        self.f6 = F6()
        self.fcoutput = FCoutput()
        
    def forward(self, img):
        
        output = self.c1(img)
        
        x = self.c3_1(output)
        output = self.c3_2(output)
        output += x
        
        output = self.c5(output)
        output = output.view(img.size(0), -1)
        output = self.f6(output)
        output = self.fcoutput(output)
        return output







