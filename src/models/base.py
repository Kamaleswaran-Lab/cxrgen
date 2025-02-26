from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())
