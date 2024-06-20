import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import tensorboardX

class SimCLR(nn.Module):
    def __init__(self, base_encoder, out_dim):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder(pretrained=False)
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))

    def forward(self, x):
        return self.encoder(x)

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat((z_i, z_j), dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positives = torch.cat((sim_i_j, sim_j_i), dim=0)
    negatives_mask = (~torch.eye(2 * batch_size, device=sim.device).bool()).float()
    negatives = sim * negatives_mask
    logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)
    labels = torch.zeros(2 * batch_size, device=logits.device, dtype=torch.long)
    return nn.CrossEntropyLoss()(logits, labels)

transform = transforms.Compose([transforms.RandomResizedCrop(96), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
stl10_train = datasets.STL10(root='./data', split='unlabeled', transform=transform, download=True)
train_loader = DataLoader(stl10_train, batch_size=256, shuffle=True, num_workers=4)

model = SimCLR(models.resnet18, 128).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = tensorboardX.SummaryWriter()

for epoch in range(50):
    model.train()
    total_loss = 0.0
    for (x_i, _), (x_j, _) in zip(train_loader, train_loader):
        x_i, x_j = x_i.cuda(), x_j.cuda()
        z_i, z_j = model(x_i), model(x_j)
        loss = nt_xent_loss(z_i, z_j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

transform_cifar100 = transforms.Compose([transforms.Resize(96), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar100_train = datasets.CIFAR100(root='./data', train=True, transform=transform_cifar100, download=True)
cifar100_test = datasets.CIFAR100(root='./data', train=False, transform=transform_cifar100, download=True)
train_loader_cifar100 = DataLoader(cifar100_train, batch_size=256, shuffle=True, num_workers=4)
test_loader_cifar100 = DataLoader(cifar100_test, batch_size=256, shuffle=False, num_workers=4)

class LinearClassifier(nn.Module):
    def __init__(self, base_encoder, out_dim):
        super(LinearClassifier, self).__init__()
        self.encoder = base_encoder(pretrained=False)
        dim_mlp = self.encoder.fc.in_features
        self.fc = nn.Linear(dim_mlp, out_dim)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)

linear_model = LinearClassifier(models.resnet18, 100).cuda()
optimizer = optim.Adam(linear_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    linear_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader_cifar100:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = linear_model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(train_loader_cifar100)
    accuracy = correct / total
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

    linear_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader_cifar100:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = linear_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    writer.add_scalar('Accuracy/test', accuracy, epoch)
writer.close()
