import os
import cv2
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)

def crop_faces(input_dir, output_dir):
    """
    Detect and crop faces from images and save them for embedding extraction.
    Input directory structure: vggface2/person_name/*.jpg
    """
    os.makedirs(output_dir, exist_ok=True)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        out_person_dir = os.path.join(output_dir, person)
        os.makedirs(out_person_dir, exist_ok=True)
        for img_file in tqdm(os.listdir(person_dir), desc=person):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img)
            if face is not None:
                out_path = os.path.join(out_person_dir, img_file)
                face_img = (face.permute(1,2,0).int().numpy())
                cv2.imwrite(out_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
