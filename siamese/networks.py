import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """PyTorch neural network used for the Siamese network. The network uses an 
    existing convolutional neural network, usually one of PyTorch's pretrained CNNs. 
    Since the CNN is passed as an argument, a network with randomly initialized or 
    pre-trained weights can be used. The network currently expects images with 
    3-channels, so 1-channel grayscale images need to be converted to 3-channel 
    using the PyTorch transform `transforms.Grayscale(num_output_channels=3)` in the 
    transform passed to `SiamesePairedDataset`.

    **Inputs**

    - **transfer_network**: A PyTorch CNN, with the fully connected layers intact, 
    and usually with pre-trained weights initialized. For example, the Siamese network 
    could be initialized with a pre-trained ResNet34 with: 
    `model_ft = SiameseNetwork(models.resnet34(pretrained=True))`. """

    def __init__(self, transfer_network):
        super().__init__()
        self.model = transfer_network

        # Take the number of input features to the fully connected layer, 
        # replace the output with the Identity so we can hook up our new
        # fully connected layer fc_layer
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