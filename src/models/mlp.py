
import torch
import torch.nn as nn

class MLPPredictor(nn.Module):
    def __init__(self, ehr_dim=100, cxr_dim=512, hidden_dims=[256, 512]):
        super().__init__()
        
        input_dim = ehr_dim + cxr_dim
        layers = []
        
        # Build MLP layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, cxr_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, ehr_data, prev_cxr):
        # Concatenate EHR data with previous CXR embedding
        x = torch.cat([ehr_data, prev_cxr], dim=1)
        return self.network(x)