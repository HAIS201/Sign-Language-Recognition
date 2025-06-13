# 필수 라이브러리 임포트 / 必要库导入
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import SignTransformer
from dataset import SignDataset
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import joblib
from collections import Counter
from utils import (
    plot_confusion_matrix, print_classification_report, seed_everything, select_topk_lightgbm, select_topk_mutual_info, select_topk_pca
)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 얼리 스토핑 클래스 정의 / 提前停止类定义
class EarlyStopping:
    def __init__(self, patience=25, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, model, path="best_model.pth"):
        score = val_acc
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping 카운트: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        print(f"모델 저장됨: {path}")

class CollateTopk:
    def __init__(self, topk_indices):
        # top-k 인덱스를 텐서로 변환하여 저장 // 将 top-k 索引转为 tensor 并保存
        self.topk_indices = torch.tensor(topk_indices, dtype=torch.long)

    def __call__(self, batch):
        fixed_len = 60  # 입력 시퀀스의 고정 길이 설정 // 设定输入序列的固定长度为60帧
        sequences, labels = zip(*batch)  # 배치에서 시퀀스와 라벨 분리 // 从batch中分别提取序列和标签
        processed_sequences = []  # 처리된 시퀀스 저장용 리스트 // 用于保存处理后序列的列表

        for s in sequences:
            if s.shape[0] > fixed_len:
                # 시퀀스가 너무 길면 자름 // 如果序列长度大于固定长度则截断
                s = s[:fixed_len, :]
            elif s.shape[0] < fixed_len:
                pad_len = fixed_len - s.shape[0]  # 필요한 패딩 길이 계산 // 计算需要补齐的长度
                pad = torch.zeros((pad_len, s.shape[1]))  # 0으로 패딩 생성 // 创建全零padding
                s = torch.cat([s, pad], dim=0)  # 패딩을 뒤에 붙임 // 将padding拼接到序列后面
            processed_sequences.append(s)  # 처리된 시퀀스를 리스트에 추가 // 把处理后的序列加入列表

        seqs = torch.stack(processed_sequences)  # 모든 시퀀스를 하나의 텐서로 묶음 // 将所有序列堆叠成一个张量
        seqs = seqs.index_select(2, self.topk_indices)  # top-k 인덱스만 선택 // 仅选择top-k特征
        return seqs, torch.tensor(labels)  # 시퀀스와 라벨을 반환 // 返回序列与标签

class CollatePCA:
    def __init__(self, pca_model):
        self.pca = pca_model
        self.fixed_len = 60

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        processed_sequences = []

        for s in sequences:
            if s.shape[0] > self.fixed_len:
                s = s[:self.fixed_len, :]
            elif s.shape[0] < self.fixed_len:
                pad_len = self.fixed_len - s.shape[0]
                pad = torch.zeros((pad_len, s.shape[1]))
                s = torch.cat([s, pad], dim=0)

            s_pca = self.pca.transform(s.numpy())  # (60, 500)
            processed_sequences.append(torch.tensor(s_pca, dtype=torch.float32))

        return torch.stack(processed_sequences), torch.tensor(labels)


# 고정 프레임 수로 시퀀스를 보정하는 함수 / 固定帧数对齐函数
def collate_fixed_length(batch):
    fixed_len = 60
    sequences, labels = zip(*batch)
    processed_sequences = []
    for s in sequences:
        if s.shape[0] > fixed_len:
            s = s[:fixed_len, :]
        elif s.shape[0] < fixed_len:
            pad_len = fixed_len - s.shape[0]
            pad = torch.zeros((pad_len, s.shape[1]))
            s = torch.cat([s, pad], dim=0)
        processed_sequences.append(s)
    return torch.stack(processed_sequences), torch.tensor(labels)

# 클래스 불균형 처리를 위한 가중치 계산 함수 / 计算类别不平衡权重的函数
def compute_class_weights(label_list, num_classes):
    count = Counter(label_list)
    total = sum(count.values())
    return torch.tensor([
        total / (num_classes * count.get(i, 1)) for i in range(num_classes)
    ], dtype=torch.float)

# 학습 루프 정의 / 训练主流程定义
def train():
    global topk_indices
    seed_everything(42)

    # 경로 및 하이퍼파라미터 설정 / 路径和超参数设置
    DATA_ROOT = r"D:\pycharm\transformer\sign_language_transformer\data\WLASL"
    TRAIN_DIR = f"{DATA_ROOT}\\MP_Data_Train"
    TEST_DIR = f"{DATA_ROOT}\\MP_Data_Test"
    TRAIN_XLSX = f"{DATA_ROOT}\\top50_train.xlsx"
    TEST_XLSX = f"{DATA_ROOT}\\top50_test.xlsx"
    batch_size = 32
    num_classes = 50
    epochs = 250
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로딩 / 加载数据集
    print("Loading dataset...")
    full_dataset = SignDataset(root_dir=TRAIN_DIR, label_file=TRAIN_XLSX, train=True)
    total_len = len(full_dataset)
    val_len = int(total_len * 0.2)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = SignDataset(root_dir=TEST_DIR, label_file=TEST_XLSX, train=False)

    # 라벨 매핑 통일 / 统一 label2index 映射
    val_dataset.dataset.label2index = full_dataset.label2index
    val_dataset.dataset.index2label = full_dataset.index2label
    test_dataset.label2index = full_dataset.label2index
    test_dataset.index2label = full_dataset.index2label

    # ====== LightGBM  ======
    #select_num = min(100, len(train_dataset))  # 최대 100개 샘플만 사용하여 특성 선택, 메모리 폭발 방지 // 最多只采样100个样本做特征选择，防止内存爆炸
    #select_indices = train_dataset.indices[:select_num] if hasattr(train_dataset, 'indices') else list(
        #range(select_num))  # 인덱스 속성 있으면 사용, 없으면 앞에서부터 // 如果有indices属性用它，否则直接取前select_num个
    #all_train_X, all_train_y = [], []  # 특성, 라벨 저장 리스트 // 用于保存特征和标签的列表
    #for idx in select_indices:
        #X, y = full_dataset[idx]  # 데이터셋에서 샘플 추출 // 从数据集中获取样本
        #all_train_X.append(X.numpy())  # 특성 저장 // 保存特征
        #all_train_y.append(y.item())  # 라벨 저장 // 保存标签
    #all_train_X = np.stack(all_train_X)  # (N, 60, 1629)로 스택 // 堆叠成(N, 60, 1629)
    #all_train_y = np.array(all_train_y)  # (N,) 배열로 변환 // 转为(N,)数组

    #topk = 500  # top-k 특성 수, 필요에 따라 조정 가능 // top-k特征数量，可根据需求调整
    #X_train_k, topk_indices = select_topk_lightgbm(all_train_X, all_train_y, k=topk)  # LightGBM으로 top-k 특성 선택 // 用LightGBM选出top-k特征
    #input_dim = topk  # 입력 차원 설정 // 设置输入维度

    select_num = min(50, len(train_dataset))
    select_indices = train_dataset.indices[:select_num] if hasattr(train_dataset, 'indices') else list(
        range(select_num))

    all_train_X = []
    for idx in select_indices:
        X, _ = full_dataset[idx]
        all_train_X.append(X.numpy())

    all_train_X = np.stack(all_train_X)  # shape: (N, 60, 1629)

    topk = 500
    _, pca = select_topk_pca(all_train_X, k=topk)
    joblib.dump(pca, "pca_model.pkl")
    print(f"PCA 모델 저장 완료 (topk: {topk})")

    input_dim = topk  # 입력 차원 설정 // 设置输入维度

    class_weights = compute_class_weights([full_dataset[i][1] for i in train_dataset.indices], num_classes)

    #collate_fn_topk = CollateTopk(topk_indices)
    collate_fn_pca = CollatePCA(pca)

    # DataLoader 정의 / 定义训练、验证、测试 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, drop_last=True, collate_fn=collate_fn_pca)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, drop_last=False, collate_fn=collate_fn_pca)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, drop_last=False, collate_fn=collate_fn_pca)

    model = SignTransformer(input_dim=input_dim, d_model=128, num_classes=num_classes).to(device)

    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=25)

    train_acc_list, val_acc_list, test_acc_list, loss_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, train_correct, train_total = 0, 0, 0
        print(f"\n[Epoch {epoch + 1}/{epochs}]")
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        print(f"Epoch Loss: {avg_loss:.4f}")
        print(f"Train Accuracy: {train_acc * 100:.2f}%")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        val_acc = correct / total
        print(f"Val Accuracy: {val_acc * 100:.2f}%")

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        loss_list.append(avg_loss)

        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)
        test_acc = test_correct / test_total
        test_acc_list.append(test_acc)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        early_stopper(val_acc, model, path="best_model.pth")
        if early_stopper.early_stop:
            print("Early stopping triggered. 학습 종료.")
            break

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint 저장됨: {ckpt_path}")

        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
        scheduler.step()

    epochs_x = range(1, len(train_acc_list) + 1)
    plt.plot(epochs_x, train_acc_list, label='Train')
    plt.plot(epochs_x, val_acc_list, label='Val')
    plt.plot(epochs_x, test_acc_list, label='Test')
    plt.legend();
    plt.title('Accuracy Curve')
    plt.savefig("accuracy_curve.png");
    plt.close()

    plt.plot(epochs_x, loss_list, label='Train Loss', color='red')
    plt.legend();
    plt.title('Loss Curve')
    plt.savefig("loss_curve.png");
    plt.close()

    print("Evaluating on test set ...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch_y.cpu().numpy())

    label_names = [test_dataset.index2label[i] for i in range(num_classes)]
    plot_confusion_matrix(all_trues, all_preds, label_names, save_path="confusion_matrix.png")
    print_classification_report(all_trues, all_preds, test_dataset.index2label)

    final_acc = accuracy_score(all_trues, all_preds)
    print(f"\n 최종 테스트 정확도 (best_model 기준): {final_acc * 100:.2f}%")

if __name__ == '__main__':
    train()
