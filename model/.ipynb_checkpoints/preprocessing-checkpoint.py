#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms,datasets


# In[17]:


from torchvision import transforms
from PIL import Image

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel)
    transforms.Resize((28, 28)),                 # Resize to 28x28
    transforms.ToTensor(),                       # Convert to tensor (scale to [0, 1])
    transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1]
])

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Apply transformations
    image_tensor = transform(image)

    # Add batch dimension (1, 1, 28, 28)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# In[18]:

# In[ ]:




