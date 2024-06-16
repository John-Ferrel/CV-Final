from datetime import datetime

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from CNN import ResNet18
from CUTMIX import cutmix
from setting import *
from VIT import ViT


def train(epoch,writer,model,train_loader,optimizer,criterion,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # CutMix implementation
        if np.random.rand() < 0.5:  # Apply CutMix with 50% probability
            inputs, target_a, target_b, lam = cutmix(inputs, targets, alpha=1.0)
            outputs = model(inputs)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            writer.add_scalar('Loss/train', running_loss / 100, epoch * len(train_loader) + i)
            running_loss = 0.0

    # acc = 100. * correct / total
    # print(f'Training Accuracy: {acc:.3f}%')
    # writer.add_scalar('Training Accuracy', acc, epoch)

def validate(epoch,writer,model,val_loader,criterion,device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    acc = 100. * correct / total
    print(f'Validation Loss: {val_loss:.3f} | Validation Accuracy: {acc:.3f}%')
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accurary/val', acc, epoch)
    return val_loss, acc

def test(model,test_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.3f}%')

def main():
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强与加载
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data/cifar-100-python', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data/cifar-100-python', train=False, download=False, transform=transform)

    # 划分验证集
    train_size = int(SPLIT_RATIO * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 定义模型
    if MODEL == 'CNN':
        model = ResNet18().to(device)
    else:
        model = ViT().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY)
    # 你的训练代码
    # 数据加载、模型初始化、优化器设置等
    num_epochs = EPOCH
    # Tensorboard
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'runs/cifar100-{current_time}-{MODEL}-{BATCH_SIZE}-{LR}-{MOMENTUM}-{WEIGHT_DECAY}-{NUM_CLASSES}-{SPLIT_RATIO}-{LR_DECAY}-{LR_DECAY_STEP}-{EPOCH}'
    writer = SummaryWriter(log_dir)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train(epoch,writer,model,train_loader,optimizer,criterion,device)
        val_loss,_ = validate(epoch,writer,model,val_loader,criterion,device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model_{MODEL}_{current_time}.pth')
        scheduler.step()

    test(model,test_loader,device)
    writer.close()

if __name__ == '__main__':
    main()


