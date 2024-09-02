import torch
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from gcn import GCN

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
device = torch.device('mps')
data_gpu = dataset.to(device)
data = dataset[0]


# Hyper parameters
num_classes = dataset.num_classes
hidden_channels = 16
input_features = dataset.num_features

model = GCN(input_features=input_features, hidden_channels=hidden_channels, num_classes=num_classes)
model.to(device)  # load the parameters ie weight to gpu
# print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()    # clear gradients
    out = model(data.x, data.edge_index)  # perform a single forward pass
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward() # derive gradients
    optimizer.step() # Update parameters based on gradients
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    prediction = out.argmax(dim= -1) # use the class with the highest probability
    test_correct = prediction[data.test_mask] == data.y[data.test_mask] # check against ground truth
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc * 100

for epoch in range(1,100):
    loss = train()
    # print(loss)
    print(f'Epoch: {epoch:03d} Loss: {loss: .4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.1f}%')