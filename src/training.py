import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models import FaceClassifier
from evaluation import evaluate_model
import os


def train_model(train_loader, val_loader, num_classes, device):
    model = FaceClassifier(embedding_dim=512, hidden_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc, _, _ = evaluate_model(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1}/10] - Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("./models", exist_ok=True)
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "hidden_dim": 256
            }
            torch.save(checkpoint, "./models/classifier_checkpoint.pth")
            print("Model saved: ./models/classifier_checkpoint.pth")

    return model
