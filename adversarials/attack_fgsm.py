import torch
import torch.nn as nn
from .attack_base import BaseMultiInputAttack


class MultiInputFGSM(BaseMultiInputAttack):
    def __init__(self, model, eps=0.1):
        super().__init__(model) # Ruft den Konstruktor der Basisklasse auf
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_cwt, x_another, labels):
        # Mache eine Kopie, um die Originaldaten nicht zu verändern
        adv_x_cwt = x_cwt.clone().detach().requires_grad_(True)
        adv_x_features = x_another.clone().detach().requires_grad_(True)

        # Berechne Loss und Gradienten
        outputs = self.model(adv_x_cwt, adv_x_features)
        loss = self.loss_fn(outputs, labels)
        self.model.zero_grad()
        loss.backward()

        # Erzeuge die adversariellen Beispiele (der FGSM-Schritt)
        with torch.no_grad():

            grad_cwt = adv_x_cwt.grad
            if grad_cwt is not None:
                # Dieser Block wird für Ihr iTransformer-Modell NICHT ausgeführt
                adv_x_cwt = adv_x_cwt + self.eps * grad_cwt.sign()

            grad_features = adv_x_features.grad
            if grad_features is not None:
                # Dieser Block wird ausgeführt, da x_features verwendet wird
                adv_x_features = adv_x_features + self.eps * grad_features.sign()

        return adv_x_cwt, adv_x_features