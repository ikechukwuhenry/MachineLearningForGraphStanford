# imports for torch geometric
import torch
# from torch.nn import Linear
# import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from pytorch_mlp_model import MLP

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
device = torch.device('mps')
data_gpu = dataset.to(device)
data = dataset[0]

print(dataset.num_features)
print(dataset.num_classes)
print(data.x)
print(data)



# Hyper parameters
num_classes = dataset.num_classes
hidden_channels = 16
input_features = dataset.num_features

model = MLP(input_features=input_features, hidden_channels=hidden_channels, num_classes=num_classes)
model.to(device)  # load the parameters ie weight to gpu
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()    # clear gradients
    out = model(data.x)  # perform a single forward pass
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward() # derive gradients
    optimizer.step() # Update parameters based on gradients
    return loss

for epoch in range(1,10):
    loss = train()
    print(loss)
    break