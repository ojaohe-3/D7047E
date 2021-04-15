import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ModelTrainer import *

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

writer = SummaryWriter()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    #==== Alex Model ====
    model = torchvision.models.alexnet(pretrained=True) #alexNet has an output of 1000 softmax linear layer.
    data = iter(trainloader)
    example, _ = data.next()
    writer.add_graph(model, example)
    
     # === FineTunning ===
    model1 = torch.nn.Sequential(model, nn.Linear(1000, 10))
    writer.add_graph(model1, example)

    model1.to(device)
    train(model1, 25, trainloader, optim.Adam(model1.parameters(), lr=0.001), title="Fine Tunning AlexNet On CIFAR-10", writer=writer)
    validate(model1, testloader)
    # === Feature Extraction Tunning ===
    model2 = torch.nn.Sequential(model, nn.Linear(1000, 10))
    # nn.Module object does not derive from PyObject so we have re download it.
    
    #disable trainning on all parameters
    for param in model2.parameters():
        param.requires_grad = False
    
    
    model2.to(device)
    writer.add_graph(model2, example)

    optimizer = optim.Adam(model2.parameters(), lr=0.001)
    train(model2, 80, trainloader, optim.Adam(model1.parameters(), lr=0.001), title="Feature Extraction AlexNet On CIFAR-10", writer=writer)
    validate(model2, testloader)
