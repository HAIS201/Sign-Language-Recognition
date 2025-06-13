import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score
import pickle


# ===== 클래스 레이블 사전 =====
actions = [ '화재', '화장실', '화요일', '화약', '화상', '홍수', '호흡기', '호흡곤란', '형', '폭발',
    '팔꿈치', '팔', '파편', '파도', '트럭', '트랙터', '통학버스', '토하다', '토요일', '택시',
    '탈골', '코', '칼', '침수', '친구', '출혈', '출산', '축사', '추락', '총',
    '체온계', '창백하다', '창문', '차안', '차밖', '집단폭행', '집', '질식', '진통제', '지혈대',
    '지난', '중랑구', '중구', '주', '종로구', '조난', '제초제', '절도', '절단', '장단지',
    '장난감', '작은방', '작년', '자상', '자살', '자동차', '임신한아내', '임산부', '일요일', '인대',
    '이웃집', '이상한사람', '이번', '이물질', '이마', '의사', '응급처리', '응급대원', '음식물', '은평구',
    '유치원 버스', '유치원', '유리', '운동장', '우리집', '용산구', '욕실', '왼쪽-눈', '왼쪽-귀', '왼쪽',
    '올해', '옥상', '오빠', '공장', '공원', '공사장', '곰', '골절', '고장', '고열',
    '고압전선', '고속도로', '계단', '계곡', '경찰차', '경찰', '경운기', '결박', '걸렸다', '가렵다'
]
classes = {label: idx for idx, label in enumerate(actions)}

# ===== 테스트셋 데이터 경로 =====
TEST_DATA_PATH = "C:/Users/2reny/Desktop/LSTM2/MP_Data_Test"
MAX_SEQ_LEN = 80

# ===== 시퀀스 데이터 읽기 =====
sequences, labels = [], []
for action in actions:
    action_path = os.path.join(TEST_DATA_PATH, action)
    if not os.path.exists(action_path): continue
    folders = sorted(os.listdir(action_path))
    for folder in folders:
        folder_path = os.path.join(action_path, folder)
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        if len(frame_files) < 40: continue
        window = []
        for f in frame_files:
            frame = np.load(os.path.join(folder_path, f))
            window.append(frame)
        sequences.append(window)
        labels.append(classes[action])


# ===== 패딩 및 라벨 처리 =====
X_test_new = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')

# ===== Feature selector 적용 =====
with open('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

X_test_2d = X_test_new.reshape(-1, X_test_new.shape[2])  # (샘플×시퀀스, 피쳐)
X_test_selected = selector.transform(X_test_2d)
X_test_fs = X_test_selected.reshape(X_test_new.shape[0], X_test_new.shape[1], -1)

y_test_new = to_categorical(labels, num_classes=len(actions))

# ===== 모델 로드 (간단해짐) =====
model = load_model("C:/Users/2reny/Desktop/LSTM2/lstm_model_final.keras")

# ===== 평가 =====
y_pred = model.predict(X_test_fs)
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test_new, axis=1)

print(f"\n[새 테스트셋 정확도] {accuracy_score(y_true_label, y_pred_label) * 100:.2f}%")
print("\n[분류 리포트]")
print(classification_report(y_true_label, y_pred_label, zero_division=0))