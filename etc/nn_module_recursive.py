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

def _modules_test(module_top, recursive=True):
    global cnt
    # Non-recursive case
    for idx, module in module_top._modules.items():
        print(module.__class__.__name__)
        if 'conv' in module.__class__.__name__.lower():
            cnt += 1        
        if recursive: # <== Recursive!
            _modules_test(module)


print('CASE 1 : Recursive w/ _modules.items()')
cnt = 0
_modules_test(model, recursive=True)        
print('# of conv layer is : ',cnt)
print('========================================')
print('CASE 2 : Non-Recursive w/ _modules.items()')
cnt = 0
_modules_test(model, recursive=False)
print('# of conv layer is : ',cnt)

def named_modules_test(module_top):
    global cnt
    # Non-recursive case
    for name, module in module_top.named_modules():
        print(module.__class__.__name__)
        if 'conv' in module.__class__.__name__.lower():
            cnt += 1        
        
print('========================================')
print('CASE 3 : None-Recursive w/ named_modules()')
cnt = 0
named_modules_test(model)
print('# of conv layer is : ',cnt)

def modules_test(module_top):
    global cnt
    # Non-recursive case
    for module in module_top.modules():
        print(module.__class__.__name__)
        if 'conv' in module.__class__.__name__.lower():
            cnt += 1   

print('========================================')
print('CASE 4 : None-Recursive w/ modules()')
cnt = 0
modules_test(model)
print('# of conv layer is : ',cnt)

# => 모든 layer를 탐색하기 위해서 _modules.items()를 사용하면 recursive 함수가 필요하지만 named_modules()나 modules()를 사용하면 recursive가 필요하지 않다.
# => check how "_modules.items()" is used in grad-CAM code (jacobgil)

# CASE 1 : Recursive w/ _modules.items()
# Sequential
# Conv2d
# Conv2d
# Dropout
# Conv2d
# Conv2d
# Linear
# # of conv layer is :  4
# ========================================
# CASE 2 : Non-Recursive w/ _modules.items()
# Sequential
# Conv2d
# Conv2d
# Linear
# # of conv layer is :  2
# ========================================
# CASE 3 : None-Recursive w/ named_modules()
# NeuralNet
# Sequential
# Conv2d
# Conv2d
# Dropout
# Conv2d
# Conv2d
# Linear
# # of conv layer is :  4
# ========================================
# CASE 4 : None-Recursive w/ modules()
# NeuralNet
# Sequential
# Conv2d
# Conv2d
# Dropout
# Conv2d
# Conv2d
# Linear
# # of conv layer is :  4