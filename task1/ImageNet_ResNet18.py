import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import tensorboardX

# 加载ImageNet数据集
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_train = datasets.ImageNet(root='./data', split='train', transform=transform_train, download=True)
imagenet_val = datasets.ImageNet(root='./data', split='val', transform=transform_test, download=True)
train_loader = DataLoader(imagenet_train, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(imagenet_val, batch_size=256, shuffle=False, num_workers=4)

# 定义ResNet-18模型
model = models.resnet18(pretrained=True, num_classes=1000).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = tensorboardX.SummaryWriter()

# 训练ResNet-18模型
for epoch in range(90):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    writer.add_scalar('Accuracy/Val', accuracy, epoch)

# 使用CIFAR-100数据集进行线性分类协议评估
transform_cifar100 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar100_train = datasets.CIFAR100(root='./data', train=True, transform=transform_cifar100, download=True)
cifar100_test = datasets.CIFAR100(root='./data', train=False, transform=transform_cifar100, download=True)
train_loader_cifar100 = DataLoader(cifar100_train, batch_size=256, shuffle=True, num_workers=4)
test_loader_cifar100 = DataLoader(cifar100_test, batch_size=256, shuffle=False, num_workers=4)

class LinearClassifier(nn.Module):
    def __init__(self, base_encoder, out_dim):
        super(LinearClassifier, self).__init__()
        self.encoder = base_encoder(pretrained=False)
        self.encoder.fc = nn.Identity()  # 去掉最后的全连接层
        dim_mlp = self.encoder.fc.in_features
        self.fc = nn.Linear(512, out_dim)  # 512是resnet18最后一层的输出维度

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
