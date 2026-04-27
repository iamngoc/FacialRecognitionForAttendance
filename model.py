# suggestion: YOLOv5 or YOLOv8 (pretrained), MobileNet SSD, ResNet
# Build a model: CNN
# CNN in combine with RNN -> for real time processing
# Paper: https://www.mdpi.com/1424-8220/25/5/1410

"""
ToDo List:
1. Importing libraries
2. Improrting and preparing datasets
3. Building the CNN model
4. Training the CNN model
5. Evaluating the CNN model
"""
import time
from pathlib import Path

#import optuna
import numpy as np  # linear algebra
import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
from PIL import Image
#from matplotlib import image as mp_image

#from matplotlib.pyplot import inferno
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
import os
import torch_directml
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import csv
import kagglehub

# hyperparameter
batch_size = 64

dropout = 0.5
epochs = 30
threshold = 0.5
learning_rate = 0.0009
weight_decay = 0.0004

temp = time.time()
recall_target = 0.95
penalty_alpha = 5.0
patience = 15

# usage of GPU
device = torch_directml.device()
print(devi
ce)

# tranform (augmentation and normalizing)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=90, translate=(0.1,0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# load dataset
dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
test_dataset = Path("datasets") / "human detection dataset"

test_datas, test_labels = [], []

for class_name in os.listdir(test_dataset):
    print(class_name) # Path
    test_set = os.path.join(test_dataset, class_name)
    for image in os.listdir(test_set):
        image_path = os.path.join(test_set, image)
        test_datas.append(image_path)
        test_labels.append(class_name)

#print(test_datas)
#print(test_labels)
# for i in range(len(test_datas)):
#     print(test_datas[i], test_labels[i])

    # for human in os.listdir(os.path.join(test_dataset, class_name)):
    #     print(human)


# define human class, because CIFAR has no human class, just boy, girl, man, woman, baby classes
human_class = ["baby", "girl", "man", "woman", "boy"]
class_names = dataset.classes

human_indices = [class_names.index(c) for c in human_class]

# print(class_names)
# print(len(class_names))
# print(human_indices)

# create a directory for checkpoint
output = Path("output") / Path(f"{temp}")
weights_dir = output / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

# Convert Label to binary ( 0 for non-human and 1 for human)
class BinaryCIFAR(Dataset):
    def __init__(self, dataset, human_indices):
        self.dataset = dataset
        self.human_indices = human_indices
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        photo, label = self.dataset[index]
        label = 1 if label in self.human_indices else 0
        return photo, torch.tensor(label, dtype=torch.float32)

class HumanDetectionDataset(Dataset):
    def __init__(self, dataset, labels, transform=None):
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image_path = self.dataset[index]
        # Important: Load images
        image = Image.open(image_path).convert("RGB")
        label = int(self.labels[index])
        #print(f"Label Typ: {type(self.labels[0])}")
        """
        Important: use transforms.ToTensor() to convert image to tensor
        
        """
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# CNN Model
class human_detect_CNN(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.imageExtractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.3),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.7),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.classicator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        x = self.imageExtractor(x)
        x = self.classicator(x)
        return x

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, output_dir=output):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)

    # loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training loss", color="#8DA0CB", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation loss", color="#E78AC3", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Training- & Validierungsverlust")
    plt.legend()
    plt.grid(alpha=0.3)

    loss_path = output_dir / f"loss_curve_{temp}.pdf"
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Accuracy curve

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Training accuracy", color="#8DA0CB", linewidth=2)
    plt.plot(epochs, val_acc, label="Validation accuracy", color="#E78AC3", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.title("Trainings- & Validierungsgenauigkeit")
    plt.legend()
    plt.grid(alpha=0.3)

    acc_path = output_dir / f"accuracy_curve_{temp}.pdf"
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved plots:\n- {loss_path}\n- {acc_path}")

def save_checkpoint (model, optimizer, epoch, train_loss, val_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(filepath))
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint (model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    print(f"Checkpoint loaded from epoch {epoch}")
    print(f"Train loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}")

    return epoch, train_loss, val_loss

def save_best_model(model, filepath, metrics=None):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'threshold': threshold
    }
    if metrics:
        save_dict.update(metrics)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, str(filepath))
    print(f"Best model saved: {filepath}")

def objective (trial, penalty_alpha=penalty_alpha):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    threshold = trial.suggest_float("threshold", 0.2, 0.5, log=True)
    # DataLoader
    train_dl = DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_datas, batch_size=batch_size)

    model = human_detect_CNN(dropout).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.BCELoss().to(device)

    # Training
    for epoch in range(10):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            logits = model(xb)
            #threshold = trial.suggest_float("threshold", 0.2, 0.6)
            preds = (torch.sigmoid(logits) > threshold).int()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    if recall >= recall_target:
        objective_value = precision
    else:
        penalty = penalty_alpha * (recall_target - recall)
        objective_value = precision - penalty
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("recall", recall)
    #trial.set_user_attr("precision", precision)

    return objective_value

def train_model (epochs, train_acc_list=[], train_loss_list=[], val_acc_list=[], val_loss_list=[]):
    train_acc, train_loss = 0, 0
    best_val_loss = float("inf")
    best_val_acc = 0

    # early_stop_counter = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            #print(type(images), type(labels))
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += ((torch.sigmoid(outputs) > threshold ) == labels).sum().item()
        train_loss = running_loss / len(train_datas)
        train_acc = correct / len(train_datas)

        val_labels = []
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += ((torch.sigmoid(outputs) > threshold) == labels).sum().item()
                val_total += images.size(0)
                val_labels.append(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")

        # save checkpoint every epoch
        checkpoint_path = weights_dir / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)

        # Early Stopping and save the best model
        print (f"Best_val_loss: {best_val_loss:.5f}, val_loss: {val_loss:.5f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            early_stop_counter = 0
            best_model_state = model.state_dict()

            best_model_path = weights_dir / f"best_model.pth"
            save_best_model(model, best_model_path, metrics={
                'best_epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_train_loss': train_loss,
                'best_train_acc': train_acc
            })
            print(f"New best model! Val loss: {best_val_loss:.5f}, Val acc: {best_val_acc:.5f}")

        else:
            early_stop_counter += 1
            if early_stop_counter >= 25:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # save final model
    final_model_path = weights_dir / f"final_model.pth"
    save_best_model(model, final_model_path, metrics={
        'final_epoch': epoch + 1,
        'final_val_loss': best_val_loss,
        'final_val_acc': best_val_acc,
        'final_train_loss': train_loss,
        'final_train_acc': train_acc
    })

    print("\n"+"="*30)
    print("Model Saving summary")
    print("="*30)
    print(f"Best model saved at: {best_model_path}")
    print(f"Best Validation loss: {best_val_loss:.5f}")
    print(f"Best Validation acc: {best_val_acc:.5f}")
    print(f"Final model saved at: {final_model_path}")
    #final_val_loss = best_val_loss
    #final_val_acc = best_val_acc
    
    # Test model with test dataset

    # Load the best model for evaluation
    print(f"Loading best model for evaluation ...")
    model.load_state_dict(best_model_state)
    print("Best model loaded!")
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # Hold a batch from DataLoader
    images_show, labels_show = next(iter(test_loader))

    with torch.no_grad():
        for images, labels in test_loader:
            #print(type(images), type(labels))
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).long()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    rows = 2
    cols = 5
    num_images = 10
    #fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    #axes = axes.flatten()

    #for i in range(num_images):
        # convert tensor into image, PyTorch uses (C, H, W) while Matplotlib uses (H, W, C)
        #img = images_show[i].permute(1, 2, 0).cpu().numpy()

        # renormalize (used mean/std 0.5)
        #img = img * 0.5 + 0.5
        # guarantee that the values are in interval [0,1]
        #img = np.clip(img, 0, 1)

        # Label String
        #true_label = "0" if labels[i] == 0 else "1"
        #pred_label = "0" if preds[i] == 0 else "1"
        #probs_val = probs[i].item()

        # visualizing
        #axes[i].imshow(img)
        #color = "green" if labels[i].item() == preds[i].item() else "red"
        #axes[i].set_title(f"Ist: {true_label}\nSoll: {pred_label}\n({probs_val:.2%})", color=color)
        #axes[i].axis('off')


    all_preds = torch.cat(all_preds).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    acc = correct / total
    print(f"\nTest Accuracy: {acc:.5f}\n")
    plt.tight_layout()
    plt.show()
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("confusion_matrix")

    output_dir = output / Path("confusion matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{temp}.pdf")
    plt.close()


    return best_val_loss, best_val_acc, train_loss_list, train_acc_list, val_loss_list, val_acc_list

binary_dataset = BinaryCIFAR(dataset, human_indices)



# 80/20 split
train_datas, temp_datas = train_test_split(binary_dataset, test_size=0.2, random_state=42)
print(len(train_datas), len(temp_datas))
val_datas, test_datas = train_test_split(temp_datas, test_size=0.5, random_state=42)

#test_datas = torch.tensor(test_datas, dtype=torch.float32)
#test_labels = torch.tensor(test_labels, dtype=torch.float32)
#test_ds = HumanDetectionDataset(test_datas, test_labels, transform=test_transform)
train_loader = DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_datas, batch_size=batch_size)
test_loader = DataLoader(dataset=test_datas, batch_size=batch_size)

#test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)



model = human_detect_CNN(dropout).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []



best_val_loss, best_val_acc, train_loss_list, train_acc_list, val_loss_list, val_acc_list = train_model(epochs, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

plot_training_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list)

# create a csv-file and write all hyperparameters and results in this file
with open('Info_Model.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['time', 'learning rate', 'dropout', 'weight decay', 'threshold', 'best val loss', 'best val acc'])
    writer.writerow([temp, learning_rate, dropout, weight_decay, threshold, best_val_loss, best_val_acc])

"""
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), )
study.optimize(objective, n_trials=40)
df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
print(df.sort_values("value", ascending=False).head())

# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_param_importances(study)
"""

