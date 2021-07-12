import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, transfer_network):
        super().__init__()
        self.model = transfer_network
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc_layer = nn.Linear(num_ftrs, 2)
        
        self.num_channels = 3
        
    def forward(self, x):
        inputs1 = x[:,:self.num_channels,:,:]
        inputs2 = x[:,self.num_channels:,:,:]
        
        outputs1 = self.model(inputs1)
        outputs2 = self.model(inputs2)
        outputs = torch.abs(outputs1 - outputs2)
        outputs = self.fc_layer(outputs)
        
        return outputs