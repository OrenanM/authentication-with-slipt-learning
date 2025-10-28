import torch
import torch.nn as nn

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10):
        super().__init__()
        
        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=5, padding='same', stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding='same', stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1)
        )
             
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out)
        return out
