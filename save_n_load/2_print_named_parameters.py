import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self): 
        super(NeuralNet, self).__init__() 
        self.sequential = nn.Sequential(nn.Conv2d(1, 32, 5), nn.Conv2d(32, 64, 5), nn.Dropout(0.3)) 
        self.layer1 = nn.Conv2d(64, 128, 5) 
        self.layer2 = nn.Conv2d(128, 256, 5) 
        self.fc = nn.Linear(256*34*34, 128) 
        
    def forward(self, x): 
        output = self.sequential(x) 
        output = self.layer1(output) 
        output = self.layer2(output) 
        output = output.view(output.size()[0], -1) 
        output = self.fc(output) 
        
        return output

model = NeuralNet()

for name, param in model.named_parameters():
    print(f'name:{name}') 
    print(type(param)) 
    print(f'param.shape:{param.shape}') 
    print(f'param.requries_grad:{param.requires_grad}') 
    print('=====')

# name:sequential.0.weight
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([32, 1, 5, 5])
# param.requries_grad:True
# =====
# name:sequential.0.bias
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([32])
# param.requries_grad:True
# =====
# name:sequential.1.weight
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([64, 32, 5, 5])
# param.requries_grad:True
# =====
# name:sequential.1.bias
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([64])
# param.requries_grad:True
# =====
# name:layer1.weight
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([128, 64, 5, 5])
# param.requries_grad:True
# =====
# name:layer1.bias
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([128])
# param.requries_grad:True
# =====
# name:layer2.weight
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([256, 128, 5, 5])
# param.requries_grad:True
# =====
# name:layer2.bias
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([256])
# param.requries_grad:True
# =====
# name:fc.weight
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([128, 295936])
# param.requries_grad:True
# =====
# name:fc.bias
# <class 'torch.nn.parameter.Parameter'>
# param.shape:torch.Size([128])
# param.requries_grad:True
# =====