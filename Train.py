import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Confusion matrix plotting

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, cm[i, j], ha="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

#main
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_root",type=str, 
                        default=r"D:\Tuan Document\NCKH TLPK\NCKH_2025\Dataset\dd_dp_dataset_canon\dd_dp_dataset_png\train_c")
    parser.add_argument("--val_root", type=str, 
                        default=r"D:\Tuan Document\NCKH TLPK\NCKH_2025\Dataset\dd_dp_dataset_canon\dd_dp_dataset_png\val_c")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    args = parser.parse_args()

    # Transform cho train/val
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Dataset
    train_dataset = datasets.ImageFolder(root=args.train_root, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=args.val_root, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Classes:", train_dataset.classes)
    print("Class_to_idx:", train_dataset.class_to_idx)

    # Logging
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Fine-tune layer4 + fc
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_acc = 0.0

    # Training
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="green")
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} | Loss {loss.item():.3f}")

        train_acc = 100 * correct / total
        writer.add_scalar("Train/Loss", train_loss/len(train_loader), epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)

        # Validation
        model.eval()
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_predictions)
        acc = accuracy_score(all_labels, all_predictions)
        plot_confusion_matrix(writer, cm, train_dataset.classes, epoch)
        writer.add_scalar("Val/Accuracy", acc, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {acc*100:.2f}%")

        # Save last
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, f"{args.trained_models}/last_cnn.pt")

        # Save best
        if acc > best_acc:
            best_acc = acc
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, f"{args.trained_models}/best_cnn.pt")
