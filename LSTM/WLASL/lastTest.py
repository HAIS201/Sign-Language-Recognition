import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ===== 클래스 레이블 사전 =====
actions = [ 'all', 'bed', 'before', 'black', 'blue', 'book', 'bowling', 'can', 'candy', 'chair',
    'clothes', 'computer', 'cool', 'cousin', 'deaf', 'dog', 'drink', 'family', 'fine', 'finish',
    'fish', 'go', 'graduate', 'hat', 'hearing', 'help', 'hot', 'kiss', 'language', 'later',
    'like', 'man', 'many', 'mother', 'no', 'now', 'orange', 'shirt', 'study', 'table',
    'tall', 'thanksgiving', 'thin', 'walk', 'what', 'white', 'who', 'woman', 'year', 'yes'
]
classes = {label: idx for idx, label in enumerate(actions)}

# ===== 테스트셋 데이터 경로 =====
TEST_DATA_PATH = "C:/Users/2reny/Desktop/LSTM3/WLASL/MP_Data_Test"
MAX_SEQ_LEN = 80

# ===== 시퀀스 데이터 읽기 =====
sequences, labels = [], []
for action in actions:
    action_path = os.path.join(TEST_DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    folders = sorted(os.listdir(action_path))
    for folder in folders:
        folder_path = os.path.join(action_path, folder)
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        #if len(frame_files) < 10:
         #   continue
        window = []
        for f in frame_files:
            frame = np.load(os.path.join(folder_path, f))
            frame = np.array(frame).reshape(-1).astype('float32')  # ✅ 안전하게 reshape
            window.append(frame)
        sequences.append(window)
        labels.append(classes[action])

# ===== 디버깅용 정보 출력 =====
print(f"총 시퀀스 수: {len(sequences)}")
if sequences:
    print(f"예시 시퀀스 길이: {len(sequences[0])}")
    print(f"예시 프레임 shape: {np.array(sequences[0][0]).shape}")

# ===== 패딩 =====
X_test_new = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
print(f"X_test_new.shape: {X_test_new.shape}")
if len(X_test_new.shape) < 3:
    raise ValueError(f"패딩 결과 shape가 잘못됨: {X_test_new.shape}")

# ===== Feature selector 적용 =====
with open('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

X_test_2d = X_test_new.reshape(-1, X_test_new.shape[2])  # (샘플×시퀀스, 피쳐)
X_test_selected = selector.transform(X_test_2d)
X_test_fs = X_test_selected.reshape(X_test_new.shape[0], X_test_new.shape[1], -1)

# ===== 라벨 처리 =====
y_test_new = to_categorical(labels, num_classes=len(actions))

# ===== 모델 로드 =====
model = load_model("C:/Users/2reny/Desktop/LSTM3/lstm_model_final.keras")

# ===== 평가 =====
y_pred = model.predict(X_test_fs)
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test_new, axis=1)

print(f"\n[새 테스트셋 정확도] {accuracy_score(y_true_label, y_pred_label) * 100:.2f}%")
print("\n[분류 리포트]")
print(classification_report(y_true_label, y_pred_label, zero_division=0))
