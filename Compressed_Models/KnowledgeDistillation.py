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

import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentFeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StudentFeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim // 2)  # Smaller hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the student model
student_model = StudentFeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

# Distillation parameters
temperature = 2.0  # Controls the softening of probabilities
alpha = 0.5  # Balances between the hard and soft targets

# Adjust the training loop for the student model
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, 28*28).requires_grad_()

        # Forward pass through the teacher model
        with torch.no_grad():
            teacher_outputs = model(images)

        # Forward pass through the student model
        student_outputs = student_model(images)

        # Calculate the loss
        loss_hard = criterion(student_outputs, labels)  # Hard target loss
        loss_soft = F.kl_div(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1),
            reduction='batchmean'
        )  # Soft target loss
        loss = alpha * loss_hard + (1 - alpha) * loss_soft

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# # Performing low-rank approximation
# def low_rank_approximation(layer, rank=10):
#     # Perform SVD on the weight of a layer
#     U, S, V = torch.svd(layer.weight.data)
#     # Approximate the weight using a lower rank
#     low_rank_weight = torch.mm(U[:, :rank], torch.mm(torch.diag(S[:rank]), V.t()[:rank, :]))
#     layer.weight.data = low_rank_weight

# low_rank_approximation(model.fc1, rank=50)  # Example for fc1 layer
# low_rank_approximation(model.fc2, rank=50)  # Example for fc2 layer

    
# TODO: Implement knowledge distillation
# Performing knowledge distillation