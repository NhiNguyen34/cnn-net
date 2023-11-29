# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fc network !
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features = input_size, out_features = 50 )
        self.fc2 = nn.Linear(in_features = 50, out_features = num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-params
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1
load_model=True

# Load Data
train_dataset = datasets.MNIST(
    root='dataset/',
    train=True, 
    transform=transforms.ToTensor(),
    download=True
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_dataset = datasets.MNIST(
    root='dataset/',
    train=False, 
    transform=transforms.ToTensor(),
    download=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion =nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




for epoch in range(num_epochs):
    
    checkpoint = {
        'state_dict': model.state_dict() ,
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, 'ck.pth.tar')    
    for batch_id, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        
        optimizer.zero_grad() # each batch gradients zero !
        loss.backward()
        optimizer.step()
        

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape(x.shape[0], -1)
            
            pred = model(x)
            _, predictions = pred.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print('Acc: {num_correct}/{num_samples}')
    
    model.train()
    return num_correct / num_samples
            
# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
        
def load_checkpoint(checkpoint):
    print('===Load===')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train Network
if load_model:
    load_checkpoint(torch.load('ck.pth.tar'))
    



