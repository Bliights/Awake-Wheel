import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# Définition d'une classe pour le modèle de classification de sommeil
class SleepClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SleepClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Geler les poids des couches convolutionnelles pré-entraînées
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Remplacer la couche dense finale pour l'adapter à la classification binaire
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Définition d'un DataLoader personnalisé pour charger et prétraiter les données
class SleepDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


# Chemin vers le répertoire contenant vos images
data_dir = "/chemin/vers/votre/repertoire"
data_paths = []  # Liste des chemins vers les images
targets = []     # Liste des étiquettes (0 pour éveil, 1 pour sommeil)

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Ajoutez ici les extensions d'image supportées
            img_path = os.path.join(root, file)
            data_paths.append(img_path)
            if "sommeil" in img_path.lower():  # Si le mot "sommeil" est présent dans le chemin de l'image, l'étiquette est 1 (sommeil)
                targets.append(1)
            else:
                targets.append(0)  # Sinon, l'étiquette est 0 (éveil)


# Division des données en ensembles d'entraînement, de validation et de test
train_paths, test_paths, train_targets, test_targets = train_test_split(data_paths, targets, test_size=0.2, random_state=42)
train_paths, val_paths, train_targets, val_targets = train_test_split(train_paths, train_targets, test_size=0.1, random_state=42)

# Définition des transformations d'images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Création des jeux de données et des DataLoaders
train_dataset = SleepDataset(train_paths, train_targets, transform)
val_dataset = SleepDataset(val_paths, val_targets, transform)
test_dataset = SleepDataset(test_paths, test_targets, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialisation du modèle
model = SleepClassifier(num_classes=2)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

# Évaluation du modèle sur l'ensemble de validation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = correct / total
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Évaluation finale sur l'ensemble de test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Sauvegarde du modèle
torch.save(model.state_dict(), 'sleep_classifier.pth')