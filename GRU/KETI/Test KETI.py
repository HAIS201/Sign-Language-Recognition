import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# 1. 테스트 데이터 불러오기
X_test = np.load(r"C:\Users\golf5\Desktop\GRU1\Preprocessed_Data\X_test_pad.npy")  # shape: (num_samples, sequence_length, input_size)
y_test = np.load(r"C:\Users\golf5\Desktop\GRU1\Preprocessed_Data\y_test.npy")  # shape: (num_samples, )
y_test_labels = np.argmax(y_test, axis=1)

# 2. 모델 로드
model = load_model(r"C:\Users\golf5\Desktop\GRU1\Models\gru_model_mixedact.h5")  # 저장된 keras 모델 경로

# 3. 예측
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# 4. 정확도 평가
acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"Test Accuracy: {acc * 100:.2f}%")
