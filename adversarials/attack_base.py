import torch
import torch.nn as nn

class BaseMultiInputAttack(nn.Module):
    """
    Eine Basisklasse für unsere benutzerdefinierten Angriffe im Stil von Torchattacks.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, x_cwt, x_another, labels):
        # Diese Methode wird von den spezifischen Angriffs-Klassen überschrieben
        raise NotImplementedError