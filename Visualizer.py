import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np


# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
# train res18 network with grad-cam
