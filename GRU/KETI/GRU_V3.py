import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Masking, GRU, Dense, Bidirectional, Dropout,
    BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import pickle
import numpy as np

# GPU 설정: 사용 가능한 GPU 목록 확인 및 메모리 증가 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"{len(physical_devices)} GPU(s) found: {physical_devices}")
else:
    print("GPU를 찾지 못했습니다. CPU로 학습합니다.")

print("Mixed Activation GRU 모델 구성 시작...")

# 데이터 로딩
data_path = r"C:\Users\golf5\Desktop\GRU1\Preprocessed_Data"
X_train = np.load(f"{data_path}/X_train_pad.npy")
X_test  = np.load(f"{data_path}/X_test_pad.npy")
y_train = np.load(f"{data_path}/y_train.npy")
y_test  = np.load(f"{data_path}/y_test.npy")

input_shape  = X_train.shape[1:]
num_classes  = y_train.shape[1]
print(f"[INFO] 입력 형태: {input_shape}, 클래스 수: {num_classes}")

# 모델 정의
inputs = Input(shape=input_shape)
x = Masking(mask_value=0.0)(inputs)

# GRU 블록들 (activation='tanh')
x = Bidirectional(GRU(128, return_sequences=True, activation='tanh', dropout=0.3, recurrent_dropout=0.3))(x)
x = BatchNormalization()(x)

x = Bidirectional(GRU(64, return_sequences=True, activation='tanh', dropout=0.3, recurrent_dropout=0.3))(x)
x = BatchNormalization()(x)

x = Bidirectional(GRU(32, return_sequences=True, activation='tanh', dropout=0.3, recurrent_dropout=0.3))(x)
x = BatchNormalization()(x)

# 시퀀스 요약
x = GlobalAveragePooling1D()(x)

# Dense 블록들 (activation='elu')
x = Dense(64, activation='elu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='elu', kernel_regularizer=l2(1e-4))(x)

# 출력층 (softmax)
outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-3))(x)

model = Model(inputs, outputs)

# 컴파일
optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['categorical_accuracy']
)

# 콜백
checkpoint = ModelCheckpoint('gru_mixedact_best.keras', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)

# 학습
print("[INFO] Mixed Activation 모델 학습 시작...\n")
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)

# 학습 이력 저장
with open('gru_mixedact_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# 시각화
plt.plot(history.history['categorical_accuracy'], label='Train')
plt.plot(history.history['val_categorical_accuracy'], label='Val')
plt.title('Mixed Activation GRU Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Mixed Activation GRU Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True); plt.show()

# 모델 저장
model.save(r"D:\archive\DATA2\Saved Models\gru_model_mixedact.keras")
model.save(r"D:\archive\DATA2\Saved Models\gru_model_mixedact.h5")
print("[INFO] Mixed Activation 모델 저장 완료")

model.summary()
