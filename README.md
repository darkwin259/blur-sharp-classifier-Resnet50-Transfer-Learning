# blur-sharp-classifier-Resnet50-Transfer-Learning
A PyTorch-based transfer learning project using ResNet50 to classify images into Blur and Sharp categories.  The model is trained with custom datasets, supports TensorBoard logging, and includes scripts for both training  and inference. This repository demonstrates practical transfer learning for image quality classification.
# Blur vs Sharp Image Classification

Sử dụng ResNet50 để phân loại ảnh thành **Blur** hoặc **Sharp**.  
Hệ thống gồm 2 phần chính:
- `train.py`: script huấn luyện, log TensorBoard, tính confusion matrix.
- `test.py`: script dự đoán ảnh đơn.

---

## 📂 Cấu trúc thư mục
```
usingModeldetectblur/
├── train.py                 # Huấn luyện
├── test.py                  # Dự đoán / inference
├── trained_models/          # Checkpoints (.pt)
├── tensorboard/             # Logs TensorBoard
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Cài đặt

```bash
git clone https://github.com/<your-username>/usingModeldetectblur.git
cd usingModeldetectblur

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

---

## 📊 Dataset

Dataset cần ở dạng **ImageFolder**: dd_dp_dataset_canon 

```
dataset/
├── Blur/
│   ├── img1.jpg
│   ├── ...
├── Sharp/
│   ├── img1.jpg
│   ├── ...
```

---

## 🚀 Huấn luyện

```bash
python train.py --data_dir dataset --epochs 20 --batch_size 32 --lr 0.001
```

Các tham số:
- `--data_dir`: đường dẫn dataset
- `--epochs`: số epoch
- `--batch_size`: batch size
- `--lr`: learning rate
- `--log_dir`: nơi lưu TensorBoard (mặc định `tensorboard/`)
- `--save_dir`: thư mục lưu checkpoint (mặc định `trained_models/`)

Xem log bằng TensorBoard:

```bash
tensorboard --logdir=tensorboard/
```

---

## 🧪 Dự đoán

Chạy:

```bash
python test.py
```

Trong `test.py`, chỉnh `img_path` thành đường dẫn ảnh cần dự đoán.  
Ví dụ output:

```
Predicted: Sharp | Scores: Blur=0.123, Sharp=0.877
```

---

## 📈 Đánh giá

Script huấn luyện sẽ tính **confusion matrix** và **accuracy** bằng `sklearn.metrics`.

Ví dụ confusion matrix:

```
[[50   2]
 [ 5  43]]
```

---

## 📌 Thư viện cần thiết
Xem chi tiết trong [requirements.txt](requirements.txt).

---

## 📝 License
MIT License
