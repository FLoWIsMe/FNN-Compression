# Loading the trained model
import torch
import torch.nn as nn
from ..Control_Model import FeedforwardNeuralNetModel
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.utils.prune as prune
import torch.nn.functional as F


# Assuming FeedforwardNeuralNetModel class is defined here

# Initialize the model
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

# Load the model state
model_load_path = 'FeedforwardNeuralNetModel.pth'
model.load_state_dict(torch.load(model_load_path))

# Set the model to evaluation mode if you are not training it further
model.eval()

# Now, you can use model for inference or further processing
