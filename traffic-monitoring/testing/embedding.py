import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


def euclidean_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2, p=2).item()


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


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


if __name__ == '__main__':
    model = EfficientNetEmbedding()
    model.eval()

    image1 = load_image("img.png")
    image2 = load_image("img1.png")

    with torch.no_grad():
        embedding1 = model(image1)
        embedding2 = model(image2)

    print(euclidean_distance(embedding1, embedding2))

