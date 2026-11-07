import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import glob, os
from sklearn.metrics import precision_recall_fscore_support

META = r"models_cnn\fer4_cnn_meta.json"
CKPT = r"models_cnn\fer4_cnn_best.pt"
DATASET_ROOT = r"C:\Users\carla\PyCharmMiscProject\fer-2013"
SPLIT = "test"  # test split

with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)

classes = meta["classes"]
sad_idx = classes.index("sad")
img_size = meta["img_size"]
arch = meta["arch"]

# Build model (match training)
def make_model(arch, num_classes):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError("Add other arch if you used it")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = make_model(arch, len(classes)).to(device)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# Load test images for the 4 classes
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")
paths, y_true = [], []
for ci, c in enumerate(classes):
    folder = os.path.join(DATASET_ROOT, SPLIT, c)
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True)
    paths += files
    y_true += [ci] * len(files)

y_true = np.array(y_true, dtype=np.int64)

# Predict probabilities
probs = []
with torch.no_grad():
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        logits = model(x)
        pr = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs.append(pr)
probs = np.stack(probs, axis=0)  # (N,4)

# Threshold sweep for SAD
print("theta  sad_precision  sad_recall  sad_f1   sad_TP  sad_FP")
for theta in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    # Predict SAD only when sad prob >= theta, otherwise NOT_SAD
    pred_sad = (probs[:, sad_idx] >= theta).astype(int)
    true_sad = (y_true == sad_idx).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(true_sad, pred_sad, average="binary", zero_division=0)
    TP = int(((pred_sad == 1) & (true_sad == 1)).sum())
    FP = int(((pred_sad == 1) & (true_sad == 0)).sum())
    print(f"{theta:0.2f}    {prec:0.3f}         {rec:0.3f}     {f1:0.3f}   {TP:5d}  {FP:5d}")