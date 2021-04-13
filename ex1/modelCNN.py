import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import CNNLeReLu
import CNNTanh

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images


# leakyRelU
class CNNLeReLu (nn.Module):
    def __init__(self):
        super(CNNLeReLu, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # x = self.pool(F.leaky_relu(self.conv1(x)))
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = x.view(-1, 16 * 5 * 5)
        x = torch.flatten(x,start_dim = 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Tanh
def train(net, optimizer, criterion, title, writer):
    for epoch in range(10):  
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                writer.add_scalar('training loss, ' + title, running_loss/2000, epoch)
                writer.add_scalar('accuracy, '+title, running_correct/2000, epoch)
                
                running_loss = 0.0
                running_correct = 0.0

    print('Finished Training')

def validate(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))



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
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    train(net, optimizer, "SGD LeakyReLu")
    print("with optimizer SGD and Leaky ReLu:")
    validate(net)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    train(net, optimizer, "Adam LeakyReLu")
    print("with optimizer Adam and Leaky ReLu:")
    validate(net)

    # ==== With tanh ====
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    train(net, optimizer, "SGD tanh")
    print("with optimizer SGD and tanh:")
    validate(net)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    train(net, optimizer, "Adam tanh")
    print("with optimizer Adam and tanh:")
    validate(net)
