from __future__ import annotations

import torch

# Variable privée
_GLOBAL_DEVICE = None

def set_default_device(device_string):
    """
    Permet à l'utilisateur de forcer un appareil pour toute la session.
    Ex: antstorch.set_default_device('cpu') ou ('cuda:1')
    """
    global _GLOBAL_DEVICE
    _GLOBAL_DEVICE = torch.device(device_string)

def get_default_device():
    """
    Renvoie l'appareil défini par l'utilisateur, ou détecte automatiquement
    le meilleur accélérateur matériel disponible (CUDA -> MPS -> CPU).
    """
    global _GLOBAL_DEVICE
    if _GLOBAL_DEVICE is not None:
        return _GLOBAL_DEVICE
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_user_default_device():
    """
    Renvoie l'appareil fixé explicitement via ``set_default_device``, ou
    ``None`` si l'utilisateur n'en a pas fixé. Contrairement à
    ``get_default_device``, ne fait aucune détection automatique : permet aux
    outils qui ont leur propre logique de repli (p. ex. SyN sur MPS) de
    respecter un choix explicite sans le deviner.
    """
    return _GLOBAL_DEVICE
