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

    trainner1 = ModelTrainer(optim.SGD(net.parameters(), lr=0.0001), nn.CrossEntropyLoss(), trainloader, testloader)
    trainner2 = ModelTrainer(optim.Adam(net.parameters(), lr=0.0001), nn.CrossEntropyLoss(), trainloader, testloader)
    #==== running test with leaky ReLU ====
    trainner1.train(net,10, "SGD LeakyReLu", writer)
    print("with optimizer SGD and Leaky ReLu:")
    trainner1.validate(net)
    

    trainner2.train(net,10, "Adam LeakyReLu", writer)
    print("with optimizer Adam and Leaky ReLu:")
    trainner2.validate(net)

    # ==== With tanh ====
    trainner1.train(net1,10, "SGD tanh", writer)
    print("with optimizer SGD and tanh:")
    trainner1.validate(net1)
    
    trainner2.train(net1,10, "Adam tanh", writer)
    print("with optimizer Adam and tanh:")
    trainner2.validate(net1)

