import torch.nn as nn
from .module import ResnetBlock2n 


class ResNet_16_cifar(nn.Module): 

    def __init__(self, dropout=0.1, nb_classes=100, *args, **kwargs): 

        super().__init__(*args, **kwargs)
        self.block1 = ResnetBlock2n(3, 64)
        self.block2 = ResnetBlock2n(64, 128, reduce_dim=True)
        self.block3 = ResnetBlock2n(128, 256, reduce_dim=True)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(4096, nb_classes)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x): 

        bs = x.shape[0]
        x1 = self.block1(x) 
        x2 = self.block2(x1) 
        x3 = self.block3(x2)
        out = self.pool(x3) 
        out = self.dropout(out)
        out = out.view(bs, -1)
        out = self.fc(out) 

        return out


    
        