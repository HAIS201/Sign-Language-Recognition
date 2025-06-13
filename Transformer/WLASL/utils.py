import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif

# 프레임 단위 정규화(z-score) / 帧级 z-score 归一化
def normalize_frames(frames):
    """
    각 프레임 내 특징을 평균 0, 분산 1로 정규화 / 对每帧内部特征做 z-score 归一化
    frames: list[np.ndarray] 또는 (T, D)
    반환: 동일 shape, 정규화된 프레임들
    """
    frames = np.array(frames)
    mean = np.mean(frames, axis=1, keepdims=True)
    std = np.std(frames, axis=1, keepdims=True) + 1e-6
    return (frames - mean) / std

# 혼동 행렬 시각화 함수 / 可视化并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, figsize=(12, 10)):
    """
    혼동 행렬을 히트맵으로 출력 / 绘制热力图形式的混淆矩阵
    y_true, y_pred: 예측값과 실제값 / 预测值与真实标签
    labels: 클래스 이름 리스트 / 类别名称列表
    save_path: 저장 경로 (png 파일) / 图片保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 분류 평가 리포트 출력 / 输出分类评估报告
def print_classification_report(y_true, y_pred, labels):
    """
    Precision, Recall, F1-score 리포트 출력 / 打印分类的准确率、召回率和F1分数
    """
    def print_classification_report(y_true, y_pred, index2label):

        used_labels = sorted(list(set(y_true) | set(y_pred)))  # 실제 등장한 클래스만 추림
        label_names = [index2label[i] for i in used_labels]

        report = classification_report(y_true, y_pred,
                                    target_names=label_names,
                                    labels=used_labels,
                                    digits=4)
        print(report)

# 랜덤 시드 고정 / 固定随机种子，保证结果可复现
def seed_everything(seed=42):
    """
    실험 재현성을 위한 시드 고정 / 固定全局随机种子
    """
    import random, os
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def select_topk_lightgbm(X, y, k=500):
    """
    사용 LightGBM 특성 중요도로 판별력 높은 k개 특성 선택 // 用LightGBM特征重要性选最优k个特征
    """
    X_flat = X.reshape(-1, X.shape[-1])  # 시계열 평탄화 // 展平时间序列
    y_flat = np.repeat(y, X.shape[1])  # 라벨 반복 // 标签重复
    lgb_model = lgb.LGBMClassifier(n_estimators=10, n_jobs=2, random_state=42)  # LightGBM 모델 생성 // 创建LightGBM模型
    lgb_model.fit(X_flat, y_flat)  # 모델 학습 // 模型训练
    importances = lgb_model.feature_importances_  # 특성 중요도 추출 // 获取特征重要性
    topk_indices = np.argsort(importances)[-k:]  # 중요도 상위 k개 인덱스 선택 // 选择重要性最高的k个索引
    X_k = X[:, :, topk_indices]  # 선택된 특성만 추출 // 仅保留选中特征
    return X_k, topk_indices  # 결과 반환 // 返回结果

def select_topk_mutual_info(X, y, k=500):
    """
    互信息法选取top-k特征 / MI 방법으로 top-k 특징 선택
    X: np.ndarray, shape (N, T, D)  # N: 샘플 수, T: 프레임 수, D: 특성 차원
    y: np.ndarray, shape (N,)       # N: 샘플 수
    k: int                          # 선택할 top-k 특징 수
    返回:
        X_k: (N, T, k)              # top-k 特征后的新数据
        topk_indices: 被选中特征的下标 (특징 인덱스)
    """


    N, T, D = X.shape
    X_flat = X.reshape(-1, D)      # (N*T, D) 모든 프레임을 행렬로 합침
    y_rep = np.repeat(y, T)        # (N*T,) 각 프레임에 대해 라벨 반복

    # 计算每个特征的互信息分数 / 각 특징별 MI 점수 계산
    mi = mutual_info_classif(X_flat, y_rep, discrete_features=False)
    topk_indices = np.argsort(mi)[-k:]  # top-k 특징 인덱스 선택

    X_k = X[:, :, topk_indices]         # top-k 특징만 남김
    return X_k, topk_indices

def select_topk_pca(X, k=500):
    """
    PCA를 사용하여 입력 데이터 X의 주요 k개 특성을 선택
    X: np.ndarray, shape (N, T, D)
    반환:
        X_k: (N, T, k)
        pca_model: PCA 모델 객체 (transform 시 활용 가능)
    """
    N, T, D = X.shape
    X_flat = X.reshape(N * T, D)  # (N*T, D)
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X_flat)  # (N*T, k)
    X_k = X_reduced.reshape(N, T, k)  # 다시 (N, T, k)로
    return X_k, pca

