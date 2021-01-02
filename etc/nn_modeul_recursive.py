import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self): 
        super(NeuralNet, self).__init__() 
        self.sequential = nn.Sequential(nn.Conv2d(1, 32, 5), nn.Conv2d(32, 64, 5), nn.Dropout(0.3)) 
        self.conv1 = nn.Conv2d(64, 128, 5) 
        self.conv2 = nn.Conv2d(128, 256, 5) 
        self.fc = nn.Linear(256*34*34, 128) 
        
    def forward(self, x): 
        output = self.sequential(x) 
        output = self.conv1(output) 
        output = self.conv2(output) 
        output = output.view(output.size()[0], -1) 
        output = self.fc(output) 
        
        return output

model = NeuralNet()

def recursive_relu_apply(module_top, recursive=True):
    global cnt
    # Non-recursive case
    for idx, module in module_top._modules.items():
        print(module.__class__.__name__)
        if 'conv' in module.__class__.__name__.lower():
            cnt += 1        
        if recursive:
            recursive_relu_apply(module) # <== Recursive


print('CASE 1 : Non-Recursive')
cnt = 0
recursive_relu_apply(model, recursive=False)
print('Total # of conv in network is : ',cnt)
print('========================================')
print('CASE 2 : Recursive')
cnt = 0
recursive_relu_apply(model, recursive=True)        
print('Total # of conv in network is : ',cnt)

# CASE 1 : Non-Recursive
# Sequential
# Conv2d
# Conv2d
# Linear
# Total # of conv in network is :  2
# ========================================
# CASE 2 : Recursive
# Sequential
# Conv2d
# Conv2d
# Dropout
# Conv2d
# Conv2d
# Linear
# Total # of conv in network is :  4