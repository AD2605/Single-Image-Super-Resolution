from AutoEncoder import AutoEncoder
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(1234)
import cv2

data_HR = DataLoader(
    torchvision.datasets.ImageFolder(
        '/home/atharva/Datasets/DIV2K_train_HR/',
        transform=transforms.Compose([
            transforms.Resize((1440, 2048)),
            transforms.ToTensor()
        ])
    ),
    num_workers=2,
    batch_size=1,
    shuffle=False,
    pin_memory=False
)
data_LR = DataLoader(
    torchvision.datasets.ImageFolder(
        '/home/atharva/Datasets/DIV2K_4x/',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ),
    num_workers=2,
    batch_size=1,
    shuffle=False,
    pin_memory=False
)
val_HR = DataLoader(
    torchvision.datasets.ImageFolder(
        '/home/atharva/Datasets/DIV2K_VAL_HR',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ),
    num_workers=2,
    batch_size=1,
    shuffle=False,
    pin_memory=False
)
val_LR = DataLoader(
    torchvision.datasets.ImageFolder(
        '/home/atharva/Datasets/DIV2K_VAL_LR',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ),
    num_workers=2,
    batch_size=1,
    shuffle=False,
    pin_memory=False
)
'''
for (x,y), (x_d, y_d) in zip(data_HR, data_LR):
    print(x.shape)
    x = x.permute(2, 3, 1, 0)
    x = x.squeeze().detach().cpu().numpy()
    x_d = x_d.permute(2, 3, 1, 0)
    x_d = x_d.squeeze().detach().cpu().numpy()
    cv2.imshow('image', x)
    cv2.imshow('degraded', x_d)
    cv2.waitKey(delay=3000)
'''
model = AutoEncoder()
model.train_model(dataloader_HR=data_HR, dataloader_LR=data_LR, model=model, epochs=75, val_HR=val_HR, val_LR=val_LR)
