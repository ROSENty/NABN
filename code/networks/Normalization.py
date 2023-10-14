import torch, pdb
from torch import nn
import numpy as np

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, num_dims, momentum=0.9, eps=1e-5):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.momentum = momentum
        self.eps = eps

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        if not torch.is_grad_enabled():
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                mean = X.mean(dim=0).detach()
                var = ((X - mean)**2).mean(dim=0).detach()
            else:
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
            
            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            self.moving_mean = (self.momentum * self.moving_mean + (1.0 - self.momentum) * mean).detach()
            self.moving_var = (self.momentum * self.moving_var + (1.0 - self.momentum) * var).detach()
            
        Y = self.gamma * X_hat + self.beta
        return Y


class GauBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_dims, theta=0.2, momentum=0.9, eps=1e-5):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.momentum = momentum
        self.eps = eps
        self.theta = theta
        print('self.theta is ', self.theta)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        if not torch.is_grad_enabled():
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                mean = X.mean(dim=0)
                var = ((X - mean)**2).mean(dim=0)
            else:
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                
                mean_normal = torch.normal(mean=mean, std=torch.ones_like(mean))
                mean_normal = torch.maximum(mean_normal, -self.theta+mean)
                mean_normal = torch.minimum(mean_normal, self.theta+mean)
                mean = mean_normal

                var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)

            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            self.moving_mean = (self.momentum * self.moving_mean + (1.0 - self.momentum) * mean).detach()
            self.moving_var = (self.momentum * self.moving_var + (1.0 - self.momentum) * var).detach()
            
        Y = self.gamma * X_hat + self.beta  # Scale and shift
        return Y
    
class GroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, X):
        
        N, C, H, W = X.size()
        X = X.view(N, self.num_groups, -1)

        mean = X.mean(dim=(2), keepdim=True)
        var = ((X - mean)**2).mean(dim=(2), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + self.eps)
            
        X_hat = X_hat.view(N, C, H, W)
        
        Y = self.gamma * X_hat + self.beta  # Scale and shift
        return Y
    
class UNBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_dims, theta=0.3, momentum=0.9, eps=1e-5):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.momentum = momentum
        self.eps = eps
        self.theta = theta
        print('self.theta is ', self.theta)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        if not torch.is_grad_enabled():
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                mean = X.mean(dim=0)
                var = ((X - mean)**2).mean(dim=0)
            else:
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                
                mean_uniform = self.theta * 2 * torch.rand_like(mean) + (mean-self.theta)
                
                mean = mean_uniform

                var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            self.moving_mean = (self.momentum * self.moving_mean + (1.0 - self.momentum) * mean).detach()
            self.moving_var = (self.momentum * self.moving_var + (1.0 - self.momentum) * var).detach()
            
        Y = self.gamma * X_hat + self.beta  # Scale and shift
        return Y