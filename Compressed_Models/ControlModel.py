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




# # Pruning the FeedforwardNeuralNetModel
# def prune_model(model, pruning_rate=0.3):
#     # Pruning 30% of the connections in both linear layers
#     prune.l1_unstructured(model.fc1, name='weight', amount=pruning_rate)
#     prune.l1_unstructured(model.fc2, name='weight', amount=pruning_rate)
#     # To make pruning permanent
#     prune.remove(model.fc1, 'weight')
#     prune.remove(model.fc2, 'weight')

# prune_model(model)


# Appyling Quantization
import torch.quantization

# Dynamic Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Specify which layers to quantize
    dtype=torch.qint8
)

# Performing low-rank approximation
def low_rank_approximation(layer, rank=10):
    # Perform SVD on the weight of a layer
    U, S, V = torch.svd(layer.weight.data)
    # Approximate the weight using a lower rank
    low_rank_weight = torch.mm(U[:, :rank], torch.mm(torch.diag(S[:rank]), V.t()[:rank, :]))
    layer.weight.data = low_rank_weight

low_rank_approximation(model.fc1, rank=50)  # Example for fc1 layer
low_rank_approximation(model.fc2, rank=50)  # Example for fc2 layer

    
# TODO: Implement knowledge distillation
# Performing knowledge distillation