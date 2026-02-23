import torch
import torch.nn as nn


class FaceClassifier(nn.Module):
    def __init__(self, input_dim=524, hidden_dim=512, num_classes=480):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class HybridFusionClassifier(nn.Module):
    """Trainable fusion module for embedding + geometry branches."""

    def __init__(self, emb_dim=512, geo_dim=12, hidden_dim=256, num_classes=480):
        super().__init__()
        self.emb_dim = emb_dim
        self.geo_dim = geo_dim
        self.emb_branch = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.geo_branch = nn.Sequential(nn.Linear(geo_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = x[:, : self.emb_dim]
        geo = x[:, self.emb_dim : self.emb_dim + self.geo_dim]
        emb_h = self.emb_branch(emb)
        geo_h = self.geo_branch(geo)
        alpha = self.gate(torch.cat([emb_h, geo_h], dim=1))
        fused = alpha * emb_h + (1.0 - alpha) * geo_h
        return self.head(fused)


def combine_features(embedding: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    return torch.cat([embedding, geometry], dim=-1)
