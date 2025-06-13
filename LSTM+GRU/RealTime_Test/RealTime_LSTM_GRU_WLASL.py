import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque
import time

# 모델과 PCA 로드
model = tf.keras.models.load_model(r"C:\Users\chaeyeonhan\OneDrive\2025-1\capstone_design\models\WLASL\LSTM+GRU.keras")
pca = joblib.load(r"C:\Users\chaeyeonhan\OneDrive\2025-1\capstone_design\test\WLASL\LSTM+GRU\pca_560.joblib")

# 라벨 정의
label_map = {'all': 0, 'bed': 1, 'before': 2, 'black': 3, 'book': 4, 'bowling': 5, 'can': 6, 'candy': 7, 'chair': 8, 'clothes': 9,
             'computer': 10, 'cool': 11, 'cousin': 12, 'drink': 13, 'family': 14, 'finish': 15, 'fish': 16, 'go': 17, 'graduate': 18,
             'hat': 19, 'hearing': 20, 'help': 21, 'hot': 22, 'language': 23, 'later': 24, 'like': 25, 'man': 26, 'many': 27,
             'mother': 28, 'no': 29, 'now': 30, 'orange': 31, 'shirt': 32, 'study': 33, 'table': 34, 'tall': 35, 'thin': 36,
             'white': 37, 'who': 38, 'woman': 39, 'year': 40, 'yes': 41}
index_to_label = {v: k for k, v in label_map.items()}
classes = [index_to_label[i] for i in range(len(index_to_label))]

SEQ_LENGTH = 60
NUM_FEATURES = 560
sequence = deque(maxlen=SEQ_LENGTH)

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True)

# 🔹 정규화 함수 추가 
def normalize_single_frame(frame):
    return (frame - np.mean(frame)) / (np.std(frame) + 1e-6)

def extract_keypoints(results):
    # 얼굴: 468점만 사용
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:468]]).flatten()
    else:
        face = np.zeros(468 * 3)


    # 포즈: 33점
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:33]]).flatten()
    else:
        pose = np.zeros(33 * 3)
    

    # 왼손
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
 

    # 오른손
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
   

    # 전체 연결
    all_concat = np.concatenate([face, pose, left_hand, right_hand])

    return all_concat



# 웹캠 시작
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        keypoints = extract_keypoints(results)
        keypoints = normalize_single_frame(keypoints)                    
        keypoints_pca = pca.transform(keypoints.reshape(1, -1))[0]       
        sequence.append(keypoints_pca)

        if len(sequence) == SEQ_LENGTH:
            input_seq = np.expand_dims(sequence, axis=0)
            y_pred = model.predict(input_seq, verbose=0)
            class_idx = np.argmax(y_pred)
            confidence = y_pred[0][class_idx]
            pred_label = classes[class_idx]
        else:
            pred_label = "..."
            confidence = 0.0

        # 결과 출력
        cv2.rectangle(image, (0, 0), (300, 80), (0, 0, 0), -1)
        cv2.putText(image, f'{pred_label} ({confidence:.2f})', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 랜드마크 시각화
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Sign Recognition", image)

    except Exception as e:
        print("❌ 예외:", str(e))
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

