import torch
from typing import Union, Literal

def resolve_device(
    device: Union[str, torch.device, Literal['auto']], 
    default: torch.device
) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == 'auto':
        return default
    if device == 'cuda':
        device = 'cuda:0'
    return torch.device(device)