import torch
from torch.utils.data import DataLoader
from preprocessing import crop_faces
from feature_extraction import process_dataset
from datasets import FaceEmbeddingDataset
from training import train_model
from evaluation import evaluate_model
from sklearn.metrics import classification_report, confusion_matrix

# Directories
raw_images_dir = "data/vggface2"
cropped_dir = "data/faces_cropped"
emb_dir = "data/embeddings"

crop_faces(raw_images_dir, cropped_dir)

process_dataset(cropped_dir, emb_dir)

dataset = FaceEmbeddingDataset(emb_dir)
num_persons = len(dataset.label2idx)

train_dataset = DataLoader(dataset, batch_size=32, shuffle=True)
val_dataset = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = train_model(train_dataset, val_dataset, num_persons, device=device)

acc, all_labels, all_preds = evaluate_model(trained_model, val_dataset, device=device)

print("Overall Accuracy:", acc)
print("\nClassification Report:\n", classification_report(all_labels, all_preds))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
