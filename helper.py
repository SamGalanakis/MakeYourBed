import os
from os import listdir
from os.path import isfile, join
import matplotlib.image as img
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt




def evalOnTest(model, device, test_loader,confusion_mat=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    if confusion_mat:
        confusion_mat=confusion_matrix(target.cpu(),pred.cpu())
        
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * (correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),test_accuracy))
    return test_accuracy



def find_grayscale(path):
    images = [f for f in listdir(rf"{path}") if isfile(join(rf"{path}", f))]
    for img_path in images:
        image = img.imread(path+"//"+img_path)
        if len(image.shape)<3:
            print(img_path)
        if image.shape[-1]!=3:
            print(img_path)
find_grayscale(r"data\all_made")
find_grayscale(r"data\all_messy")