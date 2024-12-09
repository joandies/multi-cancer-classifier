import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CervicalModel(nn.Module):
    def __init__(self, num_classes):
        super(CervicalModel, self).__init__()
        
        # Load a pre-trained model (e.g., ResNet50)
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Freeze the base model's layers (so they don't get updated during training)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the final layer to match the number of classes in our dataset
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)