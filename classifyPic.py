import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.image as img
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import os
from os import listdir
from os.path import isfile, join
import urllib
from cnn_finetune import make_model
import os

def predict_image(model,url,transform):
    try:
        urllib.request.urlretrieve(url, "sample.jpg")
    except:
        print("Failed to make request to url")
        return
    try:
        image = img.imread("sample.jpg")
    except:
        print("Failed opening file")
        return
    inputImage = transform(image)
    prediction = model((inputImage.view((1,)+inputImage.shape)).to(device))
    prediction= F.softmax(prediction,dim=1).detach().cpu().numpy()
    label = np.argmax(prediction)
    os.remove("sample.jpg")
    if label==1:
        print(f"The bed is made, with probability {np.max(prediction)}")
    else:
        print(f"The bed isn't made, with probability {np.max(prediction)}")


urllib.request.urlretrieve("https://i.pinimg.com/originals/10/1b/0d/101b0daa8bf7ce15e369153d4d3ddbc9.jpg", "sample.jpg")
image = img.imread("sample.jpg")

means=np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])
img_size=224


transform= transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=(img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means,std)])







device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = make_model('resnet101', num_classes=2, pretrained=False)

checkpoint=torch.load(r"finalModel\12_model_5.tar")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model=model.to(device)

inputImage = transform(image)

prediction = model((inputImage.view((1,)+inputImage.shape)).to(device))
prediction= F.softmax(prediction,dim=1).detach().cpu().numpy()
label = np.argmax(prediction)

if label==1:
    print(f"The bed is made, with probability {np.max(prediction)}")
else:
     print(f"The bed isn't made, with probability {np.max(prediction)}")


while True:
    url=input("Give me bed url!: ")
    predict_image(model,url,transform)
print("Done")