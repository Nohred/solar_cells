import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18, ResNet18_Weights


from models.base import BaseNN

class resnet18(BaseNN):
    def __init__(self, num_classes=2):
        super(resnet18, self).__init__(name='resnet18')
        # Crear modelo y cargar pesos preentrenados
        self.network = torchvision_resnet18(weights=ResNet18_Weights.DEFAULT)

        for parameter in self.network.parameters():
            parameter.requires_grad = False 
            # Congela los pesos de las capas convolucionales para que 
            # no se actualicen durante el entrenamiento

        # Obtiene el número de características de la última capa
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def configure_trainable(self, phase="head_only"):
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        for parameter in self.network.fc.parameters():
            parameter.requires_grad = True
        if phase == "finetune":
            for parameter in self.network.layer4.parameters():
                parameter.requires_grad = True
        elif phase != "head_only":
            raise ValueError("phase must be 'head_only' or 'finetune'")

        return ["network.fc"] if phase == "head_only" else ["network.layer4", "network.fc"]

    def forward(self, x):
        return self.network(x)