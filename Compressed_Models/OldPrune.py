import torch
import torch.nn as nn

from ControlModel import FeedforwardNeuralNetModel
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# Correct the paths for loading model configuration and state
model_config_path = "configElements.pth"
model_state_path = "FeedforwardNeuralNetModel.pth"

# Loading the config elements:
checkpoint = torch.load(model_config_path)

# Extract model configuration
model_config = checkpoint['model_config']

# Initialize the model with loaded configuration
model = FeedforwardNeuralNetModel(**model_config)

# Load the model state
model.load_state_dict(torch.load(model_state_path))

# Set the model to evaluation mode if you are not training it further
model.eval()
