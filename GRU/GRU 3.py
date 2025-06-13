import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. 데이터 로딩 및 변환
# -------------------------------
X_train = np.load(r"C:\Users\golf5\Desktop\WLASL-2000\Xy_Data2\X_train.npy")
y_train = np.load(r"C:\Users\golf5\Desktop\WLASL-2000\Xy_Data2\y_train.npy")
X_test = np.load(r"C:\Users\golf5\Desktop\WLASL-2000\Xy_Data2\X_test.npy")
y_test = np.load(r"C:\Users\golf5\Desktop\WLASL-2000\Xy_Data2\y_test.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# -------------------------------
# 2. 가우시안 노이즈 함수 정의
# -------------------------------
def add_gaussian_noise(data, mean=0.0, std=0.01):
    noise = torch.randn_like(data) * std + mean
    return data + noise

# -------------------------------
# 3. 데이터 증강 (노이즈 추가)
# -------------------------------
X_train_noisy = add_gaussian_noise(X_train)
y_train_noisy = y_train.clone()

# 원본 + 노이즈 데이터 합치기
X_train_aug = torch.cat([X_train, X_train_noisy], dim=0)
y_train_aug = torch.cat([y_train, y_train_noisy], dim=0)

# -------------------------------
# 4. 데이터 로더 설정
# -------------------------------
train_dataset = TensorDataset(X_train_aug, y_train_aug)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------------
# 5. 모델 정의
# -------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, 128, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128*2, 64, batch_first=True, bidirectional=True)
        self.gru3 = nn.GRU(64*2, 32, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(32*2)
        self.fc1 = nn.Linear(32*2, 512)
        self.elu1 = nn.ELU()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.fc1(out)
        out = self.elu1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.fc2(out)
        return out

# -------------------------------
# 6. 학습 설정
# -------------------------------
INPUT_SIZE = 1629
NUM_CLASSES = len(torch.unique(y_train))
EPOCHS = 500
LEARNING_RATE = 1e-4
num_epochs = 500

model = GRUModel(INPUT_SIZE, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay = 1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# TensorBoard 설정
log_dir = r"C:\Users\golf5\Desktop\WLASL-2000\tensorboard2_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

train_accuracies = []
val_accuracies = []

# Early stopping 설정
patience = 25
trigger_times = 0
best_acc = 0.0

# 학습
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    # 검증
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = correct / total
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"[{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Best model 저장
    if val_acc > best_acc:
        best_acc = val_acc
        trigger_times = 0
        torch.save(model.state_dict(), r"C:\Users\golf5\Desktop\WLASL-2000\best_model3.pt")
        print(f"Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
    else:
        trigger_times += 1
        print(f"EarlyStopping counter: {trigger_times}/{patience}")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

    scheduler.step()

writer.close()

epochs_range = range(len(train_accuracies))
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()