import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(net, epochs, trainloader, optimizer, criterion=nn.CrossEntropyLoss(), title="", writer=None):
    print(
        f"training {title} network for {epochs} epochs, {'tensorboard enabled' if writer else 'no tensorboard enabled'}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
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
                if writer:
                    writer.add_scalar(
                        'training loss, ' + title, running_loss/2000, epoch)
                    writer.add_scalar('accuracy, '+title,
                                        running_correct/2000, epoch)

                running_loss = 0.0
                running_correct = 0.0

    print('Finished Training')

def validate(net, testloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("testing network:")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("total accuracy of net: %.2f%%" % (correct/total*100))
