import torch
import torch.nn as nn


class FaceClassifier(nn.Module):
    """Classifier over [embedding(512) + geometry(5)] features."""

    def __init__(self, input_dim=517, hidden_dim=512, num_classes=480):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def combine_features(embedding: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    return torch.cat([embedding, geometry], dim=-1)
