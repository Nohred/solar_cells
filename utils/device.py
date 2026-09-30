import torch
from functools import cache

@cache # decorador de cache
# como el dispositivo no cambia, hace que solo se ejecute una vez y ya se queda ahi en cache
def get_device(): # Indica dispositivo a usar
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def clear_device_cache():
    get_device.cache_clear()
