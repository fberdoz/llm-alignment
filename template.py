
# Test
import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

# Get the project name
PROJECT_NAME = os.path.abspath(__file__).split(os.sep)[-2]

# Parse command-line arguments to get the config file path
parser = argparse.ArgumentParser(description='Simple PyTorch Training Script with wandb and Config File')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
parser.add_argument('--taskid', type=int, default=0, help='Task ID (default: 0)')
args = parser.parse_args()

# Load configuration from the config file
with open(args.config, 'r') as f:
    config_array = yaml.safe_load(f)

# Get the specific experiment config based on the taskid
experiment_key = f"experiment_{args.taskid}"
if experiment_key in config_array:
    config = config_array[experiment_key]
else:
    raise ValueError(f"Experiment '{experiment_key}' not found in the config file.")

# Global variables
EPOCHS = config.get('num_epochs')
LR = config.get('learning_rate')
RUN_NAME = f"lr_{LR}_epochs_{EPOCHS}"

# Initialize wandb with a specific run name
wandb.init(project=PROJECT_NAME, config=config, name=RUN_NAME, mode='online')

# Print relevant information
print("Project name:", PROJECT_NAME)
print("Run name:", RUN_NAME)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_info = torch.cuda.get_device_properties(device)
    print(f"Using GPU: {gpu_info.name}")
    print(f"GPU Device ID: {torch.cuda.current_device()}")
    print(f"GPU Memory Total: {gpu_info.total_memory / (1024 ** 3):.2f} GB")
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the model and move it to the GPU if available
net = SimpleNet().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss}")

    # Log loss to wandbs
    wandb.log({'epoch': epoch + 1, 'loss': epoch_loss}, commit=True)

print("Finished Training")

# Finish the wandb run
wandb.finish()
