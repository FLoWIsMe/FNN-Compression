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
APPLY STRUCTURED SPARSITY
'''
def apply_structured_sparsity(model, sparsity_level=0.3):
    """
    Apply structured sparsity to the model, removing entire neurons based on the L1 norm of their weights.
    
    :param model: The neural network model to prune.
    :param sparsity_level: The fraction of neurons to remove (0.3 means removing 30% of neurons).
    """
    # Prune 30% of neurons in fc1 based on their weight's L1 norm
    prune.ln_structured(model.fc1, name='weight', amount=sparsity_level, n=1, dim=0)
    prune.remove(model.fc1, 'weight')  # Make the pruning permanent
    
    # Prune 30% of neurons in fc2 based on their weight's L1 norm
    # Note: For fc2, we prune input neurons, which correspond to columns in the weight matrix
    prune.ln_structured(model.fc2, name='weight', amount=sparsity_level, n=1, dim=1)
    prune.remove(model.fc2, 'weight')  # Make the pruning permanent

# Applying it to the model
apply_structured_sparsity(model)

'''
TRAIN THE MODEL
'''
import time
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0
            inference_times = []  # List to store inference times of each batch

            for images, labels in test_loader:
                images = images.view(-1, 28*28).requires_grad_()
                
                start_time = time.perf_counter()  # Start high-resolution timing
                outputs = model(images)
                end_time = time.perf_counter()  # End timing
                
                inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
                inference_times.append(inference_time)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            average_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)

            accuracy = 100 * correct / total
            print(f'Iteration: {iter}. Loss: {loss.item()}. Accuracy: {accuracy}%')
            print(f'Inference Time: Avg: {average_time:.2f} ms, Min: {min_time:.2f} ms, Max: {max_time:.2f} m')
import os

# Define the model save path
model_save_path = './Saved_Models/StructuedSparsity.pth'

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
torch.save(model.state_dict(), model_save_path)

# Save the model configuration
config_save_path = './Saved_Models/StructuredSparsityconfigElements.pth'
model_config = {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'output_dim': output_dim}
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config
}, config_save_path)
