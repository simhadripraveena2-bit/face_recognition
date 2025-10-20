import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)

            if outputs.dim() == 3:
                outputs = outputs.squeeze(1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds
