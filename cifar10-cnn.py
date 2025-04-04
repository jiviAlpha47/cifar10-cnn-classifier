import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data transforms
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define CNN
class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = CIFARClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop with accuracy tracking
for epoch in range(5):
    print(f"\nTraining Epoch {epoch + 1}...")
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i % 100 == 99:
            print(f"[Batch {i + 1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    acc = 100 * correct / total
    print(f"Epoch {epoch + 1} Training Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), 'cifar_cnn.pth')

# Load and evaluate model
model.load_state_dict(torch.load('cifar_cnn.pth'))
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Inference on custom images
custom_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = custom_transform(img).unsqueeze(0)
    return img_tensor, img

image_paths = ['frog.jpg', 'Ship.jpg', 'cat.jpeg']
for path in image_paths:
    tensor, raw_img = preprocess_image(path)
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        plt.imshow(raw_img)
        plt.title(f"Prediction: {class_names[pred.item()]}")
        plt.axis('off')
        plt.show()
