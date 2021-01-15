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

# Save entire model
torch.save(model, 'entire_model.pth')

# Load (defining model not required)
model_new = torch.load('entire_model.pth')

# print
for name, param in model_new.named_parameters():
    print(name, ':', param.requires_grad)

# sequential.0.weight : False <= 전체 모델을 저장할 경우 requires_grad 저장되었음을 확인
# sequential.0.bias : False
# sequential.1.weight : False
# sequential.1.bias : False
# layer1.weight : False
# layer1.bias : False
# layer2.weight : False
# layer2.bias : False
# fc.weight : True
# fc.bias : True