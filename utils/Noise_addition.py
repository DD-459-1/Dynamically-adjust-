

import torch

def Noise_addition(model, datasize, d_model, lr, alpha = 0.8):
    """
    Adding noise to model parameters
    Args:
            model: model that requires noise to be added.
            d_model: dimension of data variables
            lr: learning rate
    """
    temperature = 1. / (500000 * torch.sqrt(torch.tensor(d_model)))
    for name, param in model.named_parameters():
        if param.requires_grad:
            eps = torch.randn(param.size()).to(param.device)
            noise = (2.0 / lr * alpha * temperature / datasize) ** .5 * eps
            param.data.add_(noise)
    print("============Noise_addition============")