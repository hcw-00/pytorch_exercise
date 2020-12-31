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

# freeze
for name, param in model.named_parameters():
    if name in ['fc.weight','fc.bias']:
        param.requires_grad = True
    else:
        param.requires_grad = False

# print
for name, param in model.named_parameters():
    print(name, ':', param.requires_grad)

# sequential.0.weight : False
# sequential.0.bias : False
# sequential.1.weight : False
# sequential.1.bias : False
# layer1.weight : False
# layer1.bias : False
# layer2.weight : False
# layer2.bias : False
# fc.weight : True
# fc.bias : True

# Weights 만 Save
torch.save(model.state_dict(), 'weights_only.pth')

# Load
model_new = NeuralNet() # Weights 만 저장했기 때문에 모델 정의 필요
model_new.load_state_dict(torch.load('weights_only.pth'))

# print
for name, param in model_new.named_parameters():
    print(name, ':', param.requires_grad)

# sequential.0.weight : True <= Weights 만 저장할 경우 requires_grad가 저장 되지 않음!
# sequential.0.bias : True  
# sequential.1.weight : True
# sequential.1.bias : True  
# layer1.weight : True      
# layer1.bias : True
# layer2.weight : True
# layer2.bias : True
# fc.weight : True
# fc.bias : True