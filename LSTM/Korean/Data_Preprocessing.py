from modules import *
from folder_setup import *
from scipy.interpolate import interp1d
import os
import numpy as np
import collections
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# 클래스별 숫자 레이블 딕셔너리 생성
classes = {label: num for num, label in enumerate(actions)}
print(f"[클래스 매핑] → {classes}\n")

# 시퀀스 및 라벨 초기화
sequences, labels = [], []

def augment(X, y, n_augments=11):
    X_aug, y_aug = [], []

    for i in range(len(X)):
        x_orig = np.array(X[i])
        label = y[i]

        for j in range(n_augments):
            if j % 3 == 0:
                x = jitter(x_orig)
                # x = z_rotate(x)
            elif j % 3 == 1:
                # x = scale(x_orig)
                x = z_rotate(x_orig)
            else:
                x = jitter(x_orig)
                # x = scale(x)
                # x = drop_random_frames(x, drop_ratio=0.1)

            X_aug.append(np.array(x, dtype=np.float32))
            y_aug.append(label)

    return np.array(X_aug, dtype=object), np.array(y_aug)


def jitter(X, sigma=0.005):
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

"""def drop_random_frames(x, drop_ratio=0.05):
    seq_len = x.shape[0]
    n_drop = int(seq_len * drop_ratio)
    if n_drop == 0:
        return x
    drop_indices = np.random.choice(seq_len, n_drop, replace=False)
    mask = np.ones(seq_len, dtype=bool)
    mask[drop_indices] = False
    return x[mask]"""

def scale(X, scale_range=(0.98, 1.02)):
    scale_factor = np.random.uniform(*scale_range)
    return X * scale_factor

def z_rotate(X, angle_deg=10):
    theta = np.deg2rad(np.random.uniform(-angle_deg, angle_deg))
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    X_rot = X.copy()
    for i in range(X.shape[0]):
        coords = X[i].reshape(-1, 3) 
        coords_rot = coords @ rot_matrix.T
        X_rot[i] = coords_rot.reshape(-1)
    return X_rot


# 클래스별 반복
for action in actions:
    print(f"[{action}] 클래스 처리 중...")
    action_path = os.path.join(DATA_PATH, action)

    # 시퀀스 폴더 정렬
    sequence_folders = sorted(os.listdir(action_path))

    for seq_folder in sequence_folders:
        seq_path = os.path.join(action_path, seq_folder)

        # 프레임 파일 정렬
        frame_files = sorted([
            f for f in os.listdir(seq_path)
            if f.endswith('.npy')
        ])

        # 최소 프레임 필터링
        if len(frame_files) < 40:
            print(f"무시됨: {seq_path} (프레임 {len(frame_files)}개)")
            continue

        window = []
        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            res = np.load(frame_path)
            window.append(res)

        sequences.append(window)
        labels.append(classes[action])

    print(f"{action} 처리 완료 — 총 {len(sequence_folders)}개 시퀀스 중 {len([f for f in sequence_folders if len(os.listdir(os.path.join(action_path, f))) >= 50])}개 사용됨\n")

# 전체 통계 출력
print(f"전체 샘플 수: {len(sequences)}")
print(f"전체 라벨 수: {len(labels)}")

# 클래스별 샘플 수 출력 (불균형 점검)
label_counter = collections.Counter(labels)
print(f"[클래스별 샘플 분포]\n{label_counter}\n")

# numpy 배열 변환 (비정형 시퀀스 → object 타입)
X = np.array(sequences, dtype=object)
y = to_categorical(labels, num_classes=100).astype(np.float32)

# Train/Test 분리 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[전처리 전] X_train: {X_train.shape}, X_test: {X_test.shape}")

# padding 처리
MAX_SEQ_LEN = 80  # 시퀀스 길이 상한 (통계 기반 결정)

X_aug, y_aug = augment(X_train, y_train, n_augments=11)

X_train_pad = pad_sequences(X_train, padding='post', dtype='float32', maxlen=MAX_SEQ_LEN)
X_aug_pad   = pad_sequences(X_aug,   padding='post', dtype='float32', maxlen=MAX_SEQ_LEN)
X_test_pad  = pad_sequences(X_test,  padding='post', dtype='float32', maxlen=MAX_SEQ_LEN)

X_train_pad = np.concatenate((X_train_pad, X_aug_pad), axis=0)

# 유효 프레임 수 (0으로만 채워지지 않은 프레임 수) 분석
valid_lengths = [np.count_nonzero(np.any(seq != 0, axis=1)) for seq in X_train_pad]

valid_lengths = np.array(valid_lengths)
print("\n[유효 프레임 통계]")
print(f"평균 유효 프레임 수: {np.mean(valid_lengths):.2f}")
print(f"중앙값 유효 프레임 수: {np.median(valid_lengths)}")
print(f"최대 유효 프레임 수: {np.max(valid_lengths)}")
print(f"최소 유효 프레임 수: {np.min(valid_lengths)}")


y_train = np.concatenate((y_train, y_aug), axis=0)

print(f"[전처리 후] X_train_pad: {X_train_pad.shape}, X_test_pad: {X_test_pad.shape}")
print(f"[전처리 후] y_train: {y_train.shape}, y_test: {y_test.shape}")

print(f"[증강 후] X_train_pad: {X_train_pad.shape}, y_train: {y_train.shape}")

# 데이터 저장
save_path = save_path = "C:/Users/2reny/Desktop/LSTM2/PadData"

# 시퀀스 길이 분석용 원본 X도 저장
np.save(f"{save_path}/X_train.npy", X_train, allow_pickle=True)
np.save(f"{save_path}/X_test.npy", X_test, allow_pickle=True)

np.save(f"{save_path}/X_train_pad.npy", X_train_pad)
np.save(f"{save_path}/X_test_pad.npy", X_test_pad)
np.save(f"{save_path}/y_train.npy", y_train)
np.save(f"{save_path}/y_test.npy", y_test)

print("\n전처리된 데이터 저장 완료 (X_train_pad.npy, X_test_pad.npy, y_train.npy, y_test.npy)")


# pad_sequences 끝낸 후
# padding 처리된 결과를 학습용 변수에 바로 할당
"""X_train = X_train_pad
X_test = X_test_pad
y_train = y_train
y_test = y_test"""


X_train_flat = X_train_pad.reshape(-1, X_train_pad.shape[2]).astype(np.float32)
y_timewise = np.repeat(y_train.argmax(1), X_train_pad.shape[1])

selector = SelectKBest(f_classif, k=300)
X_train_selected_flat = selector.fit_transform(X_train_flat, y_timewise)
X_train_selected = X_train_selected_flat.reshape(X_train_pad.shape[0], X_train_pad.shape[1], -1)

X_test_flat = X_test_pad.reshape(-1, X_test_pad.shape[2]).astype(np.float32)
X_test_selected_flat = selector.transform(X_test_flat)
X_test_selected = X_test_selected_flat.reshape(X_test_pad.shape[0], X_test_pad.shape[1], -1)

X_train = X_train_selected
X_test = X_test_selected


with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("[INFO] feature_selector.pkl 저장 완료")