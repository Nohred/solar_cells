
from models.resnet import resnet18
from models.efficientnet import efficientnet_b0 

# patron de diseño factory - permite crear objetos sin exponer la lógica de creación 
# al cliente y referirse a la nueva instancia mediante una interfaz común
def create_model(model_name, num_classes=10):
    
    if model_name == 'resnet18':
        return resnet18(num_classes=num_classes)
    
    if model_name == 'efficientnet_b0':
        return efficientnet_b0(num_classes=num_classes)
    
    raise ValueError(f"Unknown model name: {model_name}")