import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src.evaluation import evaluate_model
from src.models import FaceClassifier


def _collect_class_prototypes(loader, num_classes):
    emb_sums = np.zeros((num_classes, 512), dtype=np.float32)
    geo_sums = np.zeros((num_classes, 5), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)

    for features, labels in loader:
        arr = features.cpu().numpy()
        lbl = labels.cpu().numpy()
        emb = arr[:, :512]
        geo = arr[:, 512:517]
        for i, c in enumerate(lbl):
            emb_sums[c] += emb[i]
            geo_sums[c] += geo[i]
            counts[c] += 1

    counts_safe = np.maximum(counts[:, None], 1)
    return emb_sums / counts_safe, geo_sums / counts_safe


def train_model(train_loader, val_loader, num_classes, device):
    model = FaceClassifier(input_dim=517, hidden_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc, _, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/20] - Loss: {running_loss / max(len(train_loader), 1):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("./models", exist_ok=True)
            emb_proto, geo_proto = _collect_class_prototypes(train_loader, num_classes)
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "hidden_dim": 256,
                "input_dim": 517,
                "embedding_prototypes": emb_proto,
                "geometry_prototypes": geo_proto,
            }
            torch.save(checkpoint, "./models/classifier_checkpoint.pth")
            print("Model saved: ./models/classifier_checkpoint.pth")

    return model
