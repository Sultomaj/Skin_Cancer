"""
train.py — Train a ResNet50 CNN with WeightedRandomSampler on the 9-class Skin Cancer ISIC Dataset.

After training completes, metrics.py is called automatically to generate all
reports and plots inside the `checkpoints/` folder.

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import kagglehub

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
NUM_CLASSES   = 9
BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LR            = 1e-4  
SAVE_DIR      = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
def build_dataloaders():
    path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
    
    train_path = os.path.join(path, "Skin cancer ISIC The International Skin Imaging Collaboration", "Train")
    test_path = os.path.join(path, "Skin cancer ISIC The International Skin Imaging Collaboration", "Test")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(root=train_path, transform=train_transforms)
    val_ds   = datasets.ImageFolder(root=test_path, transform=val_transforms)
    class_names = train_ds.classes

    # ──────────────────────────────────────────────
    # Weighted Random Sampler Setup
    # ──────────────────────────────────────────────
    class_counts = [0] * NUM_CLASSES
    for _, label in train_ds.samples:
        class_counts[label] += 1
        
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )


    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=2, 
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_ds,   
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=False
    )

    return train_loader, val_loader, class_names


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), 100 * correct / total


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, name):
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        print(f"  Epoch {epoch+1:>2}/{NUM_EPOCHS}  "
              f"Train Loss={tr_loss:.4f}  Train Acc={tr_acc:.2f}%  "
              f"Val Loss={vl_loss:.4f}  Val Acc={vl_acc:.2f}%")
    return history


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
def build_resnet50():
    """Builds and fine-tunes a ResNet50 CNN with balanced batches."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    for param in model.parameters():
        param.requires_grad = True 
        
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LR) 
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    return model, criterion, optimizer, scheduler


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    
    train_loader, val_loader, class_names = build_dataloaders()

    # ── ResNet50 ─────────────
    resnet50, criterion, optimizer, scheduler = build_resnet50()
    
    resnet50_history = run_training_loop(
        resnet50, train_loader, val_loader, criterion, optimizer, scheduler, "ResNet50_Sampler"
    )
    
    torch.save(resnet50.state_dict(), os.path.join(SAVE_DIR, "resnet50_sampler_weights.pth"))
    print(f"\nModel saved to {SAVE_DIR}/")

    # ── Auto-run metrics ─────────────────────
    import metrics
    metrics.generate_all(
        models_dict={
            "ResNet50_Sampler":  (resnet50, resnet50_history),
        },
        val_loader=val_loader,
        class_names=class_names,
        device=DEVICE,
        save_dir=SAVE_DIR
    )

if __name__ == "__main__":
    main()