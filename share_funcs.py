import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_criterion():
    """Loss をよしなに返してくれる関数"""
    return nn.CrossEntropyLoss()


def get_loaders(batch_size: int = 256, num_workers: int = 8):
    """各 Dataloader をよしなに返してくれえる関数"""
    transform = transforms.Compose([transforms.ToTensor()])

    # Dataset
    args_dataset = dict(root='./data', download=True, transform=transform)
    trainset = datasets.CIFAR10(train=True, **args_dataset)
    testset = datasets.CIFAR10(train=False, **args_dataset)

    # Data Loader
    args_loader = dict(batch_size=batch_size, num_workers=num_workers)

    train_loader = DataLoader(trainset, shuffle=True, **args_loader)
    val_loader = DataLoader(testset, shuffle=False, **args_loader)
    return train_loader, val_loader


def get_model(num_class: int = 10):
    """モデルをよしなに返してくれる関数"""
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_class)
    return model


def get_optimizer(model: torch.nn.Module, init_lr: float = 1e-3, epoch: int = 10):
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epoch*0.8), int(epoch*0.9)],
        gamma=0.1
    )
    return optimizer, lr_scheduler
