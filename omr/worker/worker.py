import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#Start TorchVision
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid

#TorchVision datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

#TorchVision transforms
from torchvision.transforms import ToTensor, Resize

import numpy as np

class CustomClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
location = "mobilenet.pth"
loaded_model = torch.load(location, map_location=torch.device('cpu'))
loaded_model.eval()
print("load")
my_classess = ['Banana peel',
 'Battery',
 'Cardboard',
 'Cloths',
 'Coffee  cup',
 'Egg shell',
 'Empty medicine packet',
 'Fish ash',
 'Glass',
 'Glaves',
 'Lemon Peel',
 'Mango Peel',
 'Mask',
 'Metal',
 'Paper',
 'Pastic',
 'Potato Peel',
 'Rice',
 'Shell of Malta',
 'Shoes',
 'Sugarcane  husk',
 'Wire']

from PIL import Image

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_image(path):
    # location = "mobilenet.pth"
    # loaded_model = torch.load(location, map_location=torch.device('cpu'))
    # loaded_model.eval()
    # print("load")
    image_path = path
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    input_data = transform(image)
    input_data = input_data.unsqueeze(0)  # Add a batch dimension

    # Move the model to the same device as the input data
    # try:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # except:
    device = torch.device("cpu")
    loaded_model.to(device)

    # Move the input data to the same device as the model
    input_data = input_data.to(device)

    # Perform inference
    with torch.no_grad():
        output = loaded_model(input_data)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    print(f"Predicted class: {predicted_class.item()}")
    return my_classess[predicted_class.item()]





# print(classify_image("test.jpg"))