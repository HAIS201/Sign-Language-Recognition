from modules import *
from keras.models import Model
from keras.layers import (Input, Masking, LSTM, Dense, Bidirectional, Activation, Permute, Multiply, Lambda, Add, Dropout, LayerNormalization, MultiHeadAttention)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.utils import register_keras_serializable
from keras.layers import Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Layer
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import tensorflow as tf
from Data_Preprocessing import *

print("[INFO] 모델 구성 시작...")

# 저장된 데이터 불러오기...
data_path = "C:/Users/2reny/Desktop/LSTM2/PadData"
# 학습 및 테스트 데이터 로딩
#X_train_pad = np.load(f"{data_path}/X_train_pad.npy")
#X_test_pad = np.load(f"{data_path}/X_test_pad.npy")
#y_train = np.load(f"{data_path}/y_train.npy")
#y_test = np.load(f"{data_path}/y_test.npy")

#X_train = X_train_pad
#X_test = X_test_pad
X_train = X_train_selected
X_test = X_test_selected

# NaN 및 무한값 확인
print("[검사] NaN/Inf 포함 여부")
print("X_train NaN:", np.isnan(X_train).sum())
print("X_test NaN:", np.isnan(X_test).sum())
print("X_train Inf:", np.isinf(X_train).sum())
print("X_test Inf:", np.isinf(X_test).sum())



# 입력 시퀀스 형태와 클래스 수 설정
input_shape = X_train_pad.shape[1:]
print(f"[INFO] 입력 형태: {input_shape}")
num_classes = y_train.shape[1] # 클래스 수

print("\n [Y 확인]")
print("Y shape:", y_train.shape)
print("Label sum axis=1 (should all be 1):", np.unique(np.sum(y_train, axis=1)))

print("\n [클래스 분포 확인]")
print("클래스별 개수 (y_train):", np.sum(y_train, axis=0))
print("클래스별 개수 (y_test):", np.sum(y_test, axis=0))

print("\n [X 정규화 확인]")
print("X_train 평균/표준편차:", np.mean(X_train), np.std(X_train))

print(f"[INFO] 입력 형태: {input_shape}")
print(f"[INFO] 클래스 수: {num_classes}")

# 입력 레이어
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Masking(mask_value=0.0)(inputs)

x1 = Bidirectional(LSTM(98, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x2 = Bidirectional(LSTM(46, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x1)
x2 = Dense(196)(x2)
x = Add()([x1, x2])

x = LayerNormalization()(x)
x = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
x = Dense(64, activation='relu', kernel_regularizer=l2(1e-3))(x)
x = Dropout(0.4)(x)
x = GlobalAveragePooling1D()(x)

x = LayerNormalization()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.4)(x)
x = LayerNormalization()(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)


#모델 컴파인
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['categorical_accuracy']
)

"""def lr_schedule(epoch, lr):
    if epoch < 10:
        return 1e-4
    elif epoch < 30:
        return 5e-5
    elif epoch < 60:
        return 1e-5
    elif epoch < 80:
        return 1e-4
    else:
        return 5e-5"""

# lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# 콜백 정의
checkpoint = ModelCheckpoint(
    'lstm_model.keras', 
    verbose=1,
    save_best_only=True,
    mode='auto'
)


# 콜백: 체크포인트 및 얼리스토핑 설정
earlystopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=10,
    cooldown=5,
    min_lr=1e-6,
    verbose=1
)

print("[INFO] 모델 구성 완료! 학습 시작...\n")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, earlystopping, reduce_lr],
    verbose=1
)


# 학습 이력 저장
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("[INFO] 학습 history 저장 완료 → training_history.pkl")

# 정확도 시각화
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.show()

# 손실 시각화
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.show()

# 최종 모델 저장
model.save('lstm_model_final.keras')
with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("[INFO] 최종 모델 저장 완료 → lstm_model_final.keras")

# 모델 요약
model.summary()
