import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.utils.prune as prune
'''
LOADING DATASET
'''

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
'''
MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

'''
CREATE MODEL CLASS
'''
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out
'''
INSTANTIATE MODEL CLASS
'''
input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

'''
INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
TRAIN THE MODEL
'''
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images with gradient accumulation capabilities
        images = images.view(-1, 28*28).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images with gradient accumulation capabilities
                images = images.view(-1, 28*28).requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


# def recompute_mask(self, theta: float = 0.001):
#     self.mask = torch.ones(
#         self.weight.shape, dtype=torch.bool, device=self.mask.device
#     )
#     self.mask[torch.where(abs(self.weight) < theta)] = False
            
# Saving the model state dictionary so I can apply compression methods to it
# Save the model
import os

# Define the model save path
model_save_path = './compressed_models/FeedforwardNeuralNetModel.pth'

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
torch.save(model.state_dict(), model_save_path)

# Save the model configuration
config_save_path = './compressed_models/configElements.pth'
model_config = {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'output_dim': output_dim}
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config
}, config_save_path)


# Appyling Quantization
import torch

# Set the quantization backend to FBGEMM for x86 architectures
torch.backends.quantized.engine = 'qnnpack'  # Use 'qnnpack' for ARM

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,  # the original model
    {nn.Linear},  # specify the layer types to quantize
    dtype=torch.qint8  # the target dtype for quantized weights
)

# Save the quantized model
quantized_model_path = './Saved_Models/QuantizedModel.pth'
torch.save(quantized_model.state_dict(), quantized_model_path)

print("Quantized model saved successfully.")

import torch
import time  # Import the time module for high-resolution timing

# Evaluate the quantized model
correct = 0
total = 0
inference_times = []  # List to store each batch's inference time

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28*28)
        
        # Start timing using perf_counter for better resolution
        start_time = time.perf_counter()
        
        outputs = quantized_model(images)
        
        # End timing
        end_time = time.perf_counter()
        
        # Calculate and store the inference time for this batch in milliseconds
        batch_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(batch_time)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate average, minimum, and maximum inference times
average_inference_time = sum(inference_times) / len(inference_times)
min_inference_time = min(inference_times)
max_inference_time = max(inference_times)

accuracy = 100 * correct / total

# Format results
formatted_accuracy = "{:.16f}".format(accuracy)
formatted_average_time = "{:.2f} ms".format(average_inference_time)
formatted_min_time = "{:.2f} ms".format(min_inference_time)
formatted_max_time = "{:.2f} ms".format(max_inference_time)

print(f'Average inference time per batch: {formatted_average_time}')
print(f'Minimum inference time per batch: {formatted_min_time}')
print(f'Maximum inference time per batch: {formatted_max_time}')
print(f'Accuracy of the quantized model on the test images: {formatted_accuracy}%')