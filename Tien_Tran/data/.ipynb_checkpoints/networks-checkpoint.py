import torch
import torch.nn as nn


class DataClassifier(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int):
        super(DataClassifier, self).__init__()
        assert input_dim == 3, "Input dimension must be 3 for RGB"
        assert output_dim > 0, "Output dimension must be a positive integer"

        self.conv1 = nn.Conv2d(
            in_channels = input_dim,
            out_channels = 16,
            kernel_size = (5, 5), 
            stride = (1, 1),
            padding = (0, 0)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (0,0)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 64, 
            kernel_size = (3, 3), 
            stride = (2, 2), 
            padding = (0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = (5,5),
            stride = (2,2),
            padding = (0,0)
        )
        self.linear1 = nn.Linear(
            in_features=64*48*48,
            out_features=output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.linear1(x)     
        x = torch.sigmoid(x)  
        return x