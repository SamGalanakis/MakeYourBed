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
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings('ignore')




# args:

seed=123
num_epochs = 1000
num_classes = 2
batch_size = 25
learning_rate = 0.001
torch.manual_seed(123)
torch.cuda.manual_seed(123)
augment= True
img_size=224
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def visualize_dataset(dataset,row=7,vert=6):
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['image.interpolation'] = 'nearest'
    mpl.rcParams['figure.figsize'] = 15, 25
    album=np.concatenate([np.concatenate([inv_normalize(dataset[i][0]).numpy() for _ in range(row)],axis=2 )for i in range(vert)],axis=1)
    plt.imshow(np.transpose(album, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    





made_files = [f for f in listdir(r"data\all_made") if isfile(join(r"data\all_made", f))]
messy_files = [f for f in listdir(r"data\all_messy") if isfile(join(r"data\all_messy", f))]

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

inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(means, std)],
   std= [1/s for s in std]
)







if augment==True:
    print("Using augmentation!")
    train_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=(img_size,img_size)),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomRotation(20),
                                        transforms.ColorJitter(.4,.4,.4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,std)])

else:
    print("Not augmenting dataset!")
    train_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=(img_size,img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,std)])





test_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=(img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means,std)])

train, test = train_test_split(df, stratify=df.Made, test_size=0.1,random_state=seed)




train_data = BedDataset(train,train_transform )

test_data = BedDataset(test, test_transform )

visualize_dataset(train_data)







train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)

test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)






model = make_model('resnet101', num_classes=2, pretrained=True)
model_name= "model_augmented"
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
print(model.original_model_info)

model=model.to(device)

train_losses = []

with open(f"models//{model_name}//{model_name}_log.txt", "a") as f:
    f.write(f"Epoch,TrainAccuracy,TrainLoss,TestAccuracy\n")
for epoch in range(1, num_epochs + 1):

    correct=0
  
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0

    
    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU 
        data = data.to(device)
        target = target.to(device)
        
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        predictions= F.softmax(output,dim=1).cpu().detach().numpy()
        predicted_labels=[np.argmax(predictions[i,:]) for i in range(target.shape[0])]
        correct += (predicted_labels == target.detach().cpu().numpy()).sum()

       
       
        train_loss += loss.item() * data.size(0)
        
  
    train_loss = train_loss/len(train_loader.sampler)
 
    train_losses.append(train_loss)
    accuracy = 100 * correct / len(train_data)
    
  
   
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, f"models//{model_name}/{epoch}_{model_name}.tar")
    
        
    
    
    print("----------------------------------------------")
    print(f"Epoch {epoch}, train_loss:{train_loss}")
    print(f"Train accuracy: {accuracy}")
    test_accuracy=evalOnTest(model,device,test_loader)
    with open(f"models//{model_name}//{model_name}_log.txt", "a") as f:
        f.write(f"{epoch},{accuracy},{train_loss},{test_accuracy} \n")
    print("----------------------------------------------")