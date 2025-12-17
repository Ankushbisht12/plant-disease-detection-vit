# ml/train_vit.py

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import PlantDiseaseDataset
from transforms import get_train_transforms, get_val_transforms
from model import build_vit_model
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "../dataset"  # PlantVillage tomato folder
SAVE_PATH = "saved_model/vit_model.pth"
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-4

os.makedirs("saved_model", exist_ok=True)

def train():
    full_dataset = PlantDiseaseDataset(
        DATASET_DIR,
        transform=get_train_transforms()
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    val_ds.dataset.transform = get_val_transforms()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_vit_model(num_classes=len(full_dataset.classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"Training batch {batch_idx}/{len(train_loader)}")

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print("Training complete. Best Val Accuracy:", best_acc)
    
    torch.save(model.state_dict(), "vit_model.pth")
    print("✅ Model saved as vit_model.pth")




if __name__ == "__main__":
    train()
