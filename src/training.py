import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_model
from src.models import HybridFusionClassifier


def _fit_scaler(train_loader):
    chunks = []
    for features, _ in train_loader:
        chunks.append(features.cpu().numpy())
    x = np.concatenate(chunks, axis=0)
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def _transform_batch(features, scaler, device):
    x = features.cpu().numpy()
    x = scaler.transform(x).astype(np.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _collect_class_prototypes(loader, num_classes, scaler, emb_dim=512, geo_dim=12):
    emb_sums = np.zeros((num_classes, emb_dim), dtype=np.float32)
    geo_sums = np.zeros((num_classes, geo_dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)

    for features, labels in loader:
        arr = scaler.transform(features.cpu().numpy()).astype(np.float32)
        lbl = labels.cpu().numpy()
        emb = arr[:, :emb_dim]
        geo = arr[:, emb_dim : emb_dim + geo_dim]
        for i, c in enumerate(lbl):
            emb_sums[c] += emb[i]
            geo_sums[c] += geo[i]
            counts[c] += 1

    counts_safe = np.maximum(counts[:, None], 1)
    return emb_sums / counts_safe, geo_sums / counts_safe


def train_model(train_loader, val_loader, num_classes, device, emb_dim=512, geo_dim=12):
    input_dim = emb_dim + geo_dim
    scaler = _fit_scaler(train_loader)
    model = HybridFusionClassifier(emb_dim=emb_dim, geo_dim=geo_dim, hidden_dim=256, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for features, labels in train_loader:
            features = _transform_batch(features, scaler, device)
            labels = labels.to(device)

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

        # scaled validation pass
        model.eval()
        all_v_preds, all_v_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = _transform_batch(features, scaler, device)
                labels = labels.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                all_v_preds.extend(preds.cpu().numpy())
                all_v_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(all_v_labels, all_v_preds)

        print(f"Epoch [{epoch + 1}/20] - Loss: {running_loss / max(len(train_loader), 1):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("./models", exist_ok=True)
            emb_proto, geo_proto = _collect_class_prototypes(train_loader, num_classes, scaler, emb_dim=emb_dim, geo_dim=geo_dim)
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "hidden_dim": 256,
                "input_dim": input_dim,
                "emb_dim": emb_dim,
                "geo_dim": geo_dim,
                "model_type": "hybrid_fusion",
                "embedding_prototypes": emb_proto,
                "geometry_prototypes": geo_proto,
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
            }
            torch.save(checkpoint, "./models/classifier_checkpoint.pth")
            print("Model saved: ./models/classifier_checkpoint.pth")

    return model
