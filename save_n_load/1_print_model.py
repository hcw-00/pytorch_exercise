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

print(model)

# NeuralNet(
#   (sequential): Sequential(
#     (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
#     (1): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
#     (2): Dropout(p=0.3, inplace=False)
#   )
#   (layer1): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
#   (layer2): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1))
#   (fc): Linear(in_features=295936, out_features=128, bias=True)
# )