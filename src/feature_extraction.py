import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = torch.tensor(np.array(img)).permute(2,0,1).float()/255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(img_tensor).cpu().numpy()
    return emb

def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        out_person_dir = os.path.join(output_dir, person)
        os.makedirs(out_person_dir, exist_ok=True)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            emb = extract_embedding(img_path)
            np.save(os.path.join(out_person_dir, img_file.replace('.jpg','.npy')), emb)
