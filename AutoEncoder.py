import torch
import torch.nn as nn
from Losses import perceptualLoss
from torchvision.transforms import transforms
import cv2
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.transform = torch.nn.functional.interpolate
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = self.conv4(out)
        out = nn.functional.relu(out)
        out = self.conv5(out)
        out = nn.functional.relu(out)
        out = self.convT1(out)
        out = nn.functional.relu(out)
        out = self.convT2(out)
        out = nn.functional.relu(out)
        out = self.convT3(out)
        out = nn.functional.relu(out)
        out = self.convT4(out)
        out = nn.functional.relu(out)
        out = self.convT5(out)
        out = self.transform(out, (1080, 1920), align_corners=False, mode='bilinear')
        return out

    def train_model(self, dataloader_HR, model, epochs, dataloader_LR, val_HR, val_LR):
        cudnn.deterministic = True
        model.cuda()
        model.train()
        optmimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = perceptualLoss()
        #mseLoss = torch.nn.MSELoss()
        scaler = amp.GradScaler()

        for epoch in range(epochs):
            print(epoch)
            model.train()
            min_loss = 1e4
            for  (x_hr, y_hr) , (x_lr, y_lr) in zip(dataloader_HR, dataloader_LR):
                optmimizer.zero_grad()
                x_lr = x_lr.cuda()
                out = model(x_lr)
                pLoss = loss(out, x_hr)
                pLoss.backward()

                #scaler.scale(pLoss).backward()
                #scaler.step(optmimizer)
                #scaler.update()

                optmimizer.step()

                x_hr = x_hr.permute(2, 3, 1, 0)
                x_hr = x_hr.squeeze().detach().cpu().numpy()
                x_lr = x_lr.permute(2, 3, 1, 0)
                x_lr = x_lr.squeeze().detach().cpu().numpy()
                out = out.permute(2, 3, 1, 0)
                out = out.squeeze().detach().cpu().numpy()
                cv2.imshow('LOW_RES',x_lr)
                cv2.imshow('GENERATED', out)
                cv2.imshow('TARGET', x_hr)
                cv2.waitKey(delay=2000)

            with torch.no_grad():
                model.eval()
                val_loss = 0
                for (x_hr, y_hr), (x_lr, y_lr) in zip(val_HR, val_LR):
                    out = model(x_lr)
                    ploss = loss(out, x_hr)
                    val_loss += ploss.item()
                    x_hr = x_hr.permute(2, 3, 1, 0)
                    x_hr = x_hr.squeeze().detach().cpu().numpy()
                    out = out.permute(2, 3, 1, 0)
                    out = out.squeeze().detach().cpu().numpy()
                    cv2.imshow('VAL_GENERATED', out)
                    cv2.imshow('VAL_TARGET', x_hr)
                    cv2.waitKey(delay=3500)
                if val_loss < min_loss:
                    torch.save(model.state_dict(), '/home/atharva/superResolution.pth')
                    min_loss = val_loss
