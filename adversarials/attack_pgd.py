import torch
import torch.nn as nn

class PGD_MultiInput(nn.Module):
    """
    Eine PGD-Angriffsklasse im Stil von Torchattacks, die explizit zwei Tensoren 
    als Input akzeptiert.
    
    Diese Klasse ist eigenständig und benötigt keine Wrapper oder externe Bibliotheken.
    """
    def __init__(self, model, eps=0.1, alpha=0.02, steps=10, random_start=True):
        """
        Args:
            model (nn.Module): Das zu attackierende Modell. Es muss eine forward-Methode
                               haben, die zwei Tensoren akzeptiert: model(x1, x2).
            eps (float): Maximale Störung (L-infinity norm).
            alpha (float): Schrittweite pro Iteration.
            steps (int): Anzahl der PGD-Iterationen.
            random_start (bool): Ob mit einer zufälligen Störung gestartet werden soll.
        """
        super().__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = next(model.parameters()).device
        # Wir verwenden die Standard Cross-Entropy Loss für die Angriffs-Generierung.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_cwt, x_features, labels):
        """
        Der eigentliche Angriffs-Loop. Nimmt zwei Tensoren entgegen.
        
        Returns:
            Ein Tupel mit den zwei adversariellen Tensoren: (adv_x_cwt, adv_x_features)
        """
        # Mache eine Kopie der Original-Inputs für die Projektion.
        x_cwt_orig = x_cwt.clone().detach()
        x_features_orig = x_features.clone().detach()
        
        # Initialisiere die Adversarial Examples mit den Original-Inputs.
        adv_x_cwt = x_cwt.clone().detach()
        adv_x_features = x_features.clone().detach()

        # Optionaler zufälliger Start
        if self.random_start:
            adv_x_cwt += torch.empty_like(adv_x_cwt).uniform_(-self.eps, self.eps)
            adv_x_features += torch.empty_like(adv_x_features).uniform_(-self.eps, self.eps)
            # Optional: Clip auf einen gültigen Datenbereich, z.B. [0, 1]
            # adv_x_cwt = torch.clamp(adv_x_cwt, min=0, max=1)
            # adv_x_features = torch.clamp(adv_x_features, min=0, max=1)

        # Die PGD-Schleife
        for _ in range(self.steps):
            # Aktiviere die Gradientenberechnung für die Inputs.
            adv_x_cwt.requires_grad = True
            adv_x_features.requires_grad = True

            # Berechne den Modell-Output. Der Aufruf ist jetzt natürlich und direkt.
            outputs = self.model(adv_x_cwt, adv_x_features)
            
            # Berechne den Loss.
            loss = self.loss_fn(outputs, labels)

            # Berechne die Gradienten des Loss bezüglich der Inputs.
            self.model.zero_grad()
            loss.backward()

            # Aktualisiere die Adversarial Examples (Gradientenaufstieg).
            with torch.no_grad():
                grad_cwt = adv_x_cwt.grad
                if grad_cwt is not None:
                    # Dieser Block wird für Ihr iTransformer-Modell NICHT ausgeführt
                    adv_x_cwt = adv_x_cwt + self.alpha * grad_cwt.sign()
                    delta = torch.clamp(adv_x_cwt - x_cwt_orig, min=-self.eps, max=self.eps)
                    adv_x_cwt = x_cwt_orig + delta
                # Wenn grad_cwt None ist, bleibt adv_x_cwt für diese Iteration unverändert.

                # Update für den zweiten Input: x_features
                grad_features = adv_x_features.grad
                if grad_features is not None:
                    # Dieser Block wird ausgeführt, da x_features verwendet wird
                    adv_x_features = adv_x_features + self.alpha * grad_features.sign()
                    delta = torch.clamp(adv_x_features - x_features_orig, min=-self.eps, max=self.eps)
                    adv_x_features = x_features_orig + delta

        # Gib das finale Adversarial Example als Tupel zurück
        return adv_x_cwt.detach(), adv_x_features.detach()