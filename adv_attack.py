import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pgd_attack(model, x_natural, y, epsilon, attack_iters, alpha, att_method, beta):
        
    x = x_natural.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)        
    for _ in range(attack_iters):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]
            
        if att_method == 'sign':
            # Use the sign function to transform the gradient
            x = x.detach() + alpha * torch.sign(grad.detach())                   
        elif att_method == 'tanh':
            # Replace the sign function with tanh
            x = x.detach() + alpha * torch.tanh(beta * grad.detach())
        elif att_method == 'raw':
            # Use the raw gradients
            x = x.detach() + alpha * grad.detach()
        elif att_method == 'norm_steep':
            # Steepest ascend direction w.r.t different norms. (i.e., G_{p}-PGD)
            grad = grad.detach()
            
            data_dim = grad.view(grad.size(0),-1).size(1)
            data_dim  = torch.Tensor([data_dim])
            beta = torch.Tensor([beta])
            step_size = torch.exp((1/data_dim) * torch.lgamma(data_dim/beta + 1) + math.log(alpha) - torch.lgamma(1/beta+1)) 
            assert not torch.isinf(step_size).any()
            assert not torch.isnan(step_size).any()
            step_size = step_size[0]
            beta = beta[0]
            data_dim = data_dim[0] 
                       
            if beta == 1: # Using coordinate gradient descend
                grad = grad.view(grad.size(0),-1)
                max_grad = torch.sign(grad) * (grad.abs() == grad.abs().max(dim=1, keepdim=True)[0])
                max_grad = max_grad.view(x.size(0), x.size(1), x.size(2), x.size(3)) 
                x = x.detach() + step_size *  max_grad          
            else:
                q = beta/(beta-1)                      
                grad_norm = torch.linalg.vector_norm(grad.view(grad.size(0),-1), ord=q, dim=1).view(-1,1,1,1)
                assert not torch.isinf(grad_norm).any()
                assert not torch.isnan(grad_norm).any()
                x = x.detach() + step_size * torch.sign(grad) * (grad.abs() / (grad_norm + 1e-10)).pow(q-1) 
        else: 
            raise ValueError
        
        x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
        assert not torch.isinf(x).any()
        assert not torch.isnan(x).any()

    return x.detach()