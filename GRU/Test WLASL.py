import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 5. GRU 모델 정의
# -------------------------------
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru1 = torch.nn.GRU(input_size, 128, batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(128*2, 64, batch_first=True, bidirectional=True)
        self.gru3 = torch.nn.GRU(64*2, 32, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(32*2)
        self.fc1 = torch.nn.Linear(32*2, 512)
        self.elu1 = torch.nn.ELU()
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, num_classes)

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
    
input_size = 1629
hidden_size = 256
num_layers = 3
num_classes = 50

model = GRUModel(input_size, num_classes).to(device)
model.load_state_dict(torch.load(r"C:\Users\golf5\Desktop\WLASL-2000\best_model3_1.pt", map_location = device))
model.eval()

# 4. 테스트 데이터 준비 (numpy -> torch.Tensor 변환)
# 예시: X_test, y_test는 numpy 배열
X_test = np.load(r'C:\Users\golf5\Desktop\WLASL-2000\Xy_Data2\X_test.npy')  # (샘플수, 시퀀스길이, input_size)
y_test = np.load(r'C:\Users\golf5\Desktop\WLASL-2000\Xy_Data2\y_test.npy')  # (샘플수, )

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 5. DataLoader로 감싸기
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 6. 정확도 계산
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')