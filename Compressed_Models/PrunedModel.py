
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import sys
sys.path.append('../Control_Model')

from Control_Model import FeedforwardNeuralNetModel


model_load_path = '/Users/ddvids123/Desktop/Thesis/Code/Compressed_Models/configElements.pth'
# Loading the config elements:
checkpoint = torch.load(model_load_path)

# Extract model configuration
model_config = checkpoint['model_config']

# Initialize the model with loaded configuration
model = FeedforwardNeuralNetModel(**model_config)

# Load the model state
model_load_path = 'Code/Compressed_Models/FeedforwardNeuralNetModel.pth'
model.load_state_dict(torch.load(model_load_path))

# Set the model to evaluation mode if you are not training it further
model.eval()

# Now, you can use model for inference or further processing
