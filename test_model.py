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
from cnn_finetune import make_model
from helper import evalOnTest



seed=123

torch.manual_seed(123)
torch.cuda.manual_seed(123)

made_files = [f for f in listdir(r"data\all_made") if isfile(join(r"data\all_made", f))]
messy_files = [f for f in listdir(r"data\all_messy") if isfile(join(r"data\all_messy", f))]
# made_files=made_files[0:2500]
min_images = min(len(made_files),len(messy_files))

zipped_cols= zip(made_files[0:min_images]+messy_files[0:min_images],[1]*min_images +[0]*min_images)
df = pd.DataFrame(zipped_cols,columns=["id","Made"])







class BedDataset(Dataset):
    def __init__(self, data, transform = None):
        super().__init__()
        self.data = data.values
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        if label==1:
            path=r"data\all_made"
        else:
            path=r"data\all_messy"
        img_path = os.path.join(path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


means=np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])
img_size=224

train_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=(img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means,std)])

test_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=(img_size,img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])




class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
    
        self.fc1 = nn.Linear(20*23*23, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

train, test = train_test_split(df, stratify=df.Made, test_size=0.1,random_state=seed)


train_data = BedDataset(train,train_transform )

test_data = BedDataset(test, test_transform )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 1000
num_classes = 2

learning_rate = 0.001
train_loader = DataLoader(dataset = train_data, batch_size = len(train_data), shuffle=True, num_workers=0)


test_loader = DataLoader(dataset = test_data, batch_size = len(test_data), shuffle=False, num_workers=0)



train_data = BedDataset(train,train_transform )

test_data = BedDataset(test, test_transform )

model=CNN()
model = make_model('resnet101', num_classes=2, pretrained=False)
checkpoint=torch.load(r"finalModel\12_model_5.tar")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model=model.to(device)

evalOnTest(model,device,test_loader)

for data,target in train_loader:
    data=data.to(device)
    target=target.to(device).cpu().numpy()

    predictions= F.softmax(model(data),dim=1).cpu().detach().numpy()
    predicted_labels=[np.argmax(predictions[i,:]) for i in range(target.shape[0])]
    accuracy = np.sum((predicted_labels==target))/target.shape[0]
    print(f"Train accuracy: {accuracy}")
print("done")
