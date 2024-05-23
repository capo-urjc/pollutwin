import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class EfficientNetEmbedding(nn.Module):
    def __init__(self):
        super(EfficientNetEmbedding, self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
                                    pretrained=True)
        self.embed = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.embed(x)
        x = torch.flatten(x, 1)

        return x
