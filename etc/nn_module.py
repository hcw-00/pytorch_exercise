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

# parameters, modules, children

print("<<named_parameters>>")
for name, param in model.named_parameters():
    print(f'name:{name}') 
    print(type(param)) 
    print('=====')

print()
print("<<named_children>>")
for name, child in model.named_children():
    print(f'name:{name}') 
    print(type(child)) 
    print('=====')

print()
print("<<named_modules>>")
for name, module in model.named_modules():
    print(f'name:{name}') 
    print(type(module)) 
    print('=====')

print()
print("<<parameters>>")
for param in model.parameters():
    print(type(param)) 
    print('=====')

print()
print("<<children>>")
for child in model.children():
    print(type(child)) 
    print('=====')

print()
print("<<modules>>")
for module in model.modules():
    print(type(module)) 
    print('=====')

print()
print("<<_parameters>>")
for param in model._parameters:
    print(type(param)) 
    print('=====')

print()
print("<<_children>>")
try:
    for child in model._children:
        print(type(child)) 
        print('=====')
except:
    print('"_children" does not exists.')

print()
print("<<_modules>>")
for module in model._modules:
    print(module)
    print(model._modules[module])
    print('=====')

# => "named_abc"와 "abc"는 name을 함께 반환하는냐를 제외하고 동일함.
# => "children"은 직계 모듈만 반환하는 반면에 "modules"는 모든 하위 모듈을 반환한다.
# => "_abc"는 Ordered dictionary를 반환한다. ("_parameter"는 비어 있고,  "_children" 은 존재하지 않는다.)

# <<named_parameters>>
# name:sequential.0.weight
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:sequential.0.bias
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:sequential.1.weight
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:sequential.1.bias
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:layer1.weight
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:layer1.bias
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:layer2.weight
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:layer2.bias
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:fc.weight
# <class 'torch.nn.parameter.Parameter'>
# =====
# name:fc.bias
# <class 'torch.nn.parameter.Parameter'>
# =====

# <<named_children>> # 직계 자식만 반환
# name:sequential
# <class 'torch.nn.modules.container.Sequential'>
# =====
# name:layer1
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# name:layer2
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# name:fc
# <class 'torch.nn.modules.linear.Linear'>
# =====

# <<named_modules>> # 하위 모든 자식을 반환
# name:
# <class '__main__.NeuralNet'>
# =====
# name:sequential
# <class 'torch.nn.modules.container.Sequential'>
# =====
# name:sequential.0
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# name:sequential.1
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# name:sequential.2
# <class 'torch.nn.modules.dropout.Dropout'>
# =====
# name:layer1
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# name:layer2
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# name:fc
# <class 'torch.nn.modules.linear.Linear'>
# =====

# <<parameters>>
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====
# <class 'torch.nn.parameter.Parameter'>
# =====

# <<children>>
# <class 'torch.nn.modules.container.Sequential'>
# =====
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# <class 'torch.nn.modules.linear.Linear'>
# =====

# <<modules>>
# <class '__main__.NeuralNet'>
# =====
# <class 'torch.nn.modules.container.Sequential'>
# =====
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# <class 'torch.nn.modules.dropout.Dropout'>
# =====
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# <class 'torch.nn.modules.conv.Conv2d'>
# =====
# <class 'torch.nn.modules.linear.Linear'>
# =====

# <<_parameters>>

# <<_children>>
# "_children" does not exists.

# <<_modules>>
# sequential
# Sequential(
#   (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
#   (1): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
#   (2): Dropout(p=0.3, inplace=False)
# )
# =====
# layer1
# Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
# =====
# layer2
# Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1))
# =====
# fc
# Linear(in_features=295936, out_features=128, bias=True)
# =====