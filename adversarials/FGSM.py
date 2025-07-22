import torch
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Adversarial:
    
    def __init__(self):
        pass
    
    def apply(self, model, x_cwt_batch, x_features_batch, labels_batch, criterion, epsilon, device):
        x_cwt_adv = x_cwt_batch.clone().detach().to(device).requires_grad_(True)
        x_features_adv = x_features_batch.clone().detach().to(device).requires_grad_(True)
        labels_adv = labels_batch.clone().detach().to(device)

        outputs = model(x_cwt_adv, x_features_adv)
        loss = criterion(outputs, labels_adv)
        model.zero_grad()
        loss.backward()

        perturbed_cwt = x_cwt_adv + epsilon * x_cwt_adv.grad.sign()
        perturbed_features = x_features_adv + epsilon * x_features_adv.grad.sign()
        return perturbed_cwt.detach(), perturbed_features.detach()
    