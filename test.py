import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

model = models.resnet50(weights=None)  
num_feature = model.fc.in_features
model.fc = nn.Linear(num_feature, 2)

checkpoint = torch.load(r"D:\Tuan Document\NCKH TLPK\NCKH_2025\8Month_AI\usingModeldetectblur\trained_models\best_cnn.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])   # load đúng state_dict đã lưu

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

class_names = ["Blur", "Sharp"] 

img_path = r"D:\Tuan Document\NCKH TLPK\NCKH_2025\Dataset\dd_dp_dataset_canon\dd_dp_dataset_png\test_c\Blur\1P0A1587.png"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# chuyển thành tensor
img_pil = transform(transforms.ToPILImage()(img_rgb)).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_pil)
    probs = torch.softmax(outputs, dim=1)[0]
    pred = torch.argmax(probs).item()

print(f"Predicted: {class_names[pred]} | "
      f"Scores: Blur={probs[0]:.3f}, Sharp={probs[1]:.3f}")


label = f"{class_names[pred]} {probs[pred]:.2f}"
cv2.putText(img_bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 2)
cv2.imshow("Prediction", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
