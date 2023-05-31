import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ResnetFeatures(nn.Module):
    def __init__(self,normalize=True,depth = False,frozen=False):
        super(ResnetFeatures, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(resnet18.children())[:-1])
        self.normalize = normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if depth:
            if frozen:
                raise Exception(f'not implemented')
            mean.append(np.mean(mean))
            std.append(np.mean(std))
            # replace the first kernel with a 4 channel version, and copy over weights for first 3  channels
            new_l1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                new_l1.weight[:,:3,:,:]= self.model[0].weight
            self.model = nn.Sequential(new_l1,*list(self.model.children())[1:])
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        self.norm_trans = transforms.Normalize(mean=mean, std=std)
    def forward(self,x):
        if self.normalize:
            # check if 0-255 or 0-1, maybe do it by checking types
            x = self.norm_trans(x)
        orig_shape = x.shape
        x = x.view(-1,*orig_shape[-3:])
        x = self.model(x)
        # reshape initial dims
        x = x.view(*orig_shape[:2],-1)
        return x

