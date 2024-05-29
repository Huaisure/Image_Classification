import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder

class CustomDataset(Dataset):
    def __init__(self,root_dir,type,transform=None):
        self.root_dir = root_dir
        csv_dir = os.path.join(root_dir,type+"_data.csv")
        self.annotations = pd.read_csv(csv_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,"imgs",self.annotations.iloc[index,0])
        image = Image.open(img_name)
        label = self.annotations.iloc[index,1]

        if self.transform:
            image = self.transform(image)

        return image,label

def get_data_loader(type:str, batch_size=32):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.RandomRotation(15),#随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#标准化，加快收敛速度
    ])

    val_transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    if type == "train":
        train_dataset = CustomDataset(data_dir,type,transform=train_transform)
        val_dataset = CustomDataset(data_dir,"val",transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        test_dataset = CustomDataset(data_dir,type,transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader