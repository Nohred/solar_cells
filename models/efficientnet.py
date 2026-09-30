import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0 as torchvision_efficientnet_b0, EfficientNet_B0_Weights


from models.base import BaseNN

class efficientnet_b0(BaseNN):
    def __init__(self, num_classes=2):
        super(efficientnet_b0, self).__init__(name='efficientnet_b0')
        # Crear modelo y cargar pesos preentrenados
        self.network = torchvision_efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for parameter in self.network.features.parameters():
            parameter.requires_grad = False 
            # Congela los pesos de las capas convolucionales para que 
            # no se actualicen durante el entrenamiento

        # Obtiene el número de características de la última capa
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, num_classes) # Reemplaza la última capa de clasificación para que tenga el número de clases deseado

    def configure_trainable(self, phase="head_only"):
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        for parameter in self.network.classifier[-1].parameters():
            parameter.requires_grad = True
        if phase == "finetune":
            for index in (7, 8):
                for parameter in self.network.features[index].parameters():
                    parameter.requires_grad = True
        elif phase != "head_only":
            raise ValueError("phase must be 'head_only' or 'finetune'")

        return ["network.classifier[-1]"] if phase == "head_only" else [
            "network.features[7]", "network.features[8]", "network.classifier[-1]"
        ]


    def forward(self, x):
        return self.network(x)