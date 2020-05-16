import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class perceptualLoss(nn.Module):
    def __init__(self):
        super(perceptualLoss, self).__init__()
        print('-------  USING VGG16 --------')
        layers = []
        layers.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        self.mse = torch.nn.MSELoss()
        for layer in layers:
            for parameters in layer:
                parameters.required_grad = False

        self.layers = nn.ModuleList(layers)
        self.transform = torch.nn.functional.interpolate

    def forward(self, generated, input):
        #generated = generated.half()
        #input = input.half()
        generated = generated.detach().cpu()
        input = input.detach().cpu()
        mse_loss = self.mse(generated, input)
        perceptual_loss = 0
        generated = self.transform(generated, mode='bilinear', size=(224, 224), align_corners=False)
        input = self.transform(input, mode='bilinear', size = (224, 224), align_corners=False)

        for layer in self.layers:
            generated = layer(generated)
            input = layer(input)
            perceptual_loss += nn.functional.l1_loss(input, generated)

        return perceptual_loss + mse_loss