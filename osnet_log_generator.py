import torch 
import torch.nn as nn
from torchreid.models.osnet import osnet_x1_0
from torchvision import transforms
import os 
import shutil
import cv2
import numpy as np
from PIL import Image
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unique_dir = "unique_people"
os.makedirs(unique_dir, exist_ok=True)
unique_embeddings = []
unique_images = []
threshold = 10.5

model = osnet_x1_0(
    num_classes = 1000,
    pretrained = False,
    loss = 'softmax'
)
state_dict = torch.load('reid_model/osnet_x1_0_msmt17.pth', map_location = device)
state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith('classifier')
}
model.load_state_dict(state_dict, strict = False)
model.to(device)
model.eval()
inf_transforms = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_img_for_inference(path):
    img = Image.open(path).convert('RGB')
    return inf_transforms(img)

def get_osnet_1x_embedding(img):
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    img_t = inf_transforms(img)
    person_crop = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        osnet_1x_embedding = model(person_crop)
    return osnet_1x_embedding

def get_avg_embedding_from_tracklet(folder_path):
    embeddings = []
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path).convert('RGB')
        with torch.no_grad():
            emb = get_osnet_1x_embedding(img)
            emb = emb.squeeze(0).cpu().numpy()
        embeddings.append(emb)
    if len(embeddings) == 0:
        return None
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def euclidean_dist(a, b):
    return np.linalg.norm(a-b)

def generate_log_from_tracklets(tracklets_dir):
    for person_folder in sorted(os.listdir(tracklets_dir)):
        folder_path = os.path.join(tracklets_dir, person_folder)

        if not os.path.isdir(folder_path):
            continue
        avg_embedding = get_avg_embedding_from_tracklet(folder_path)
        if avg_embedding is None:
            continue
        is_unique = True
        for stored_emb in unique_embeddings:
            dist = euclidean_dist(avg_embedding, stored_emb)
            if dist < threshold:
                is_unique = False
                break
        if is_unique:
            unique_embeddings.append(avg_embedding)
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    first_img_path = os.path.join(folder_path, file_name)
                    save_path = os.path.join(unique_dir, f"person_{len(unique_embeddings)}.jpg")
                    shutil.copy(first_img_path, save_path)
                    unique_images.append(save_path)
                    break
