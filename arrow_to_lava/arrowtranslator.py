import torch
from torch import nn


class ArrowTranlsator(nn.Module):
    '''
    Convet for identifying the lava locations of observations from Arrow environments
    '''

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.cnn(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, 1)

