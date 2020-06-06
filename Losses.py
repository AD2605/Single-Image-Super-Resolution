import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast

class perceptualLoss(nn.Module):
    def __init__(self):
        super(perceptualLoss, self).__init__()
        print('-------  USING VGG16 --------')
        layers = []
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[:4].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[4:9].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[9:16].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[16:23].eval())
        self.mse = torch.nn.MSELoss().cuda()
        for layer in layers:
            for parameters in layer:
                parameters.required_grad = False

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.cuda()
        for layer in self.layers:
            features = []
            with autocast():
               x = layer(x)
               features.append(x)

        return features

    def gramMatrix(self, x):
        (b, c, h ,w) = x.size()
        f = x.view(b, c, w*h)
        G = f.bmm(f.transpose(1, 2))/ (c*h*w)
        return G

    def getLoss(self, generated, target):
        generated = generated.cuda()
        target = target.cuda()
        loss = nn.MSELoss().cuda()
        target_features = self.forward(target)
        generated_features = self.forward(generated)

        #texture Loss/Style Loss
        texture_loss = 0
        generated_gram = [self.gramMatrix(feature) for feature in generated_features]
        target_gram = [self.gramMatrix(feature) for feature in target_features]

        for i in range(len(generated_gram)):
            texture_loss += loss(generated_gram[i], target_gram[i])

        #Content Loss
        target_content = target_features[1]
        generated_content = generated_features[1]
        content_loss = loss(generated_content, target_content)

        #TVL Anisotropic Denoising
        variation_x = torch.sum(torch.abs(generated[:, :, :, 1] - generated[: ,:, :, -1]))
        variation_y = torch.sum(torch.abs(generated[:, :, 1, :] - generated[:, :, -1, :]))
        TVL_loss = variation_x + variation_y
        #If using for neural style transfer, the weighted sum of the losses can be used, texture loss being with the
        #highest weight
        return texture_loss + content_loss + TVL_loss