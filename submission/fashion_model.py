import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name. 
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model does not achieve the required parameter count (101700).
    """
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(Net, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),       

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)            
        )


        # Small classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )




        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)

        x = self.features(x)          
        x = x.view(x.size(0), -1)    

        x = self.classifier(x)        
        return x
