import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ex1.CNNLeReLu import CNNLeReLu
from ex1.CNNTanh import CNNTanh
from ModelTrainer import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((1, 1, 1), (1, 1, 1))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

writer = SummaryWriter()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = CNNLeReLu()
    net1 = CNNTanh()
    
    #==== Loading tensorboard with the models
    data = iter(trainloader)
    example, _ = data.next()
    writer.add_graph(net, example)
    writer.add_graph(net1, example)

    net.to(device)
    net1.to(device)

    #==== running test with leaky ReLU ====
    train(net,10, trainloader, optim.SGD(net.parameters(), lr=0.0001), "SGD LeakyReLu", writer)
    print("with optimizer SGD and Leaky ReLu:")
    validate(net, testloader)
    

    train(net,10, trainloader, optim.Adam(net.parameters(), lr=0.0001), "Adam LeakyReLu", writer)
    print("with optimizer Adam and Leaky ReLu:")
    validate(net, testloader)

    # ==== With tanh ====
    train(net1,10, trainloader, optim.SGD(net.parameters(), lr=0.0001), "SGD tanh", writer)
    print("with optimizer SGD and tanh:")
    validate(net, testloader)
    
    train(net1,10, testloader, optim.Adam(net.parameters(), lr=0.0001), "Adam tanh", writer)
    print("with optimizer Adam and tanh:")
    validate(net, testloader)

