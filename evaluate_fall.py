import os
import cv2
import torch
import clip
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from util import cvtransform
from fall import infer, get_posture, PostureState, targets

# Adjust this path to where your dataset is located
DATASET_DIR = "./fall_archive"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
preprocess = cvtransform(preprocess)

# Prepare text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in targets]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Mapping from folder to posture state
label_map = {
    "standing": PostureState.STANDING,
    "sitting": PostureState.SITTING,
    "bending": PostureState.BENDING,
    "lying": PostureState.LYING
}

# Convert posture state to binary (Fall or Not Fall)
def binary_label(state):
    return 1 if state == PostureState.LYING else 0  # 1 = Fall, 0 = No Fall

y_true = []
y_pred = []

for label_name, true_label in label_map.items():
    folder_path = os.path.join(DATASET_DIR, label_name)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    for img_file in tqdm(image_files, desc=f"Processing {label_name}"):
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = infer(image, text_features)
        pred_label = get_posture(results)

        # y_true.append(true_label)
        # y_pred.append(pred_label)
        
        y_true.append(binary_label(true_label))
        y_pred.append(binary_label(pred_label))

# # Plot confusion matrix
# cm = confusion_matrix(y_true, y_pred, labels=[
#     PostureState.STANDING,
#     PostureState.SITTING,
#     PostureState.BENDING,
#     PostureState.LYING,
#     PostureState.UNKNOWN  # Optional: handle unknown predictions
# ])

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["Standing", "Sitting", "Bending", "Lying", "Unknown"]
# )
# disp.plot(cmap=plt.cm.Blues, values_format='d')
# plt.title("Confusion Matrix: Fall Detection")

# Compute confusion matrix (binary)
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # 1 = Fall, 0 = No Fall

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fall", "No Fall"]
)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Binary Confusion Matrix: Fall vs No Fall")
plt.tight_layout()
plt.show()

# Print classification report
print("\n" + classification_report(y_true, y_pred, target_names=["No Fall", "Fall"]))