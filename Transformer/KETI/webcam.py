import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import mediapipe as mp
import time
import os
from PIL import ImageFont, ImageDraw, Image
from utils import normalize_frames
from model import SignTransformer

class SignLanguageRecognizer:
    def __init__(self):
        # 초기화 파라미터
        self.input_dim = 500  # 모델 입력 차원
        self.seq_length = 60  # 시퀀스 길이
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_hand_landmarks = 10  # 최소 손 랜드마크 수

        # 파일 경로 설정
        self.model_path = "best_model.pth"
        self.label_path = "data/KETI_top100_clean_words.xlsx"
        self.topk_path = "topk_indices.pth"
        self.font_path = "malgun.ttf"  # 한글 폰트

        # 리소스 로드
        self._load_resources()

        # 버퍼 초기화
        self.seq_buffer = deque(maxlen=self.seq_length)
        self.pred_history = deque(maxlen=5)  # 예측 히스토리
        self.confirmed_label = None
        self.last_update_time = 0
        self.smoothing_window = 5  # 평활 윈도우 크기

        # 미디어파이프 초기화
        self._init_mediapipe()

    def _load_resources(self):
        """리소스(모델, 라벨, 특성 인덱스) 로드"""
        print("리소스 불러오는 중...")

        # 1. 라벨 로드
        try:
            df = pd.read_excel(self.label_path)
            self.index2label = {i: label for i, label in enumerate(sorted(df['한국어'].unique()))}
            self.num_classes = len(self.index2label)
            print(f"{self.num_classes}개 라벨 불러옴")
        except Exception as e:
            raise RuntimeError(f"라벨 불러오기 실패: {str(e)}")

        # 2. 특성 인덱스 로드
        try:
            self.topk_indices = torch.load(self.topk_path, map_location='cpu')
            if isinstance(self.topk_indices, torch.Tensor):
                self.topk_indices = self.topk_indices.numpy()
            print(f"{len(self.topk_indices)}차원 특성 불러옴")
        except Exception as e:
            raise RuntimeError(f"특성 인덱스 불러오기 실패: {str(e)}")

        # 3. 모델 로드
        try:
            self.model = SignTransformer(
                input_dim=self.input_dim,
                d_model=256,
                nhead=2,
                num_layers=2,
                num_classes=self.num_classes
            ).to(self.device)

            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"모델 로드 성공, 장치: {self.device}")
        except Exception as e:
            raise RuntimeError(f"모델 불러오기 실패: {str(e)}")

        # 4. 폰트 로드
        try:
            self.font = ImageFont.truetype(self.font_path, 32)
        except:
            print("경고: 한글 폰트 로드 실패, 기본 폰트 사용")
            self.font = ImageFont.load_default()

    def _init_mediapipe(self):
        """MediaPipe Holistic 초기화"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe 초기화 완료")

    def _extract_landmarks(self, results):
        """MediaPipe 결과에서 랜드마크 추출 및 정규화"""
        landmarks = []

        # 포즈 랜드마크
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)

        # 손 랜드마크
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand and len(hand.landmark) >= self.min_hand_landmarks:
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 63)

        # 얼굴 랜드마크
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 1407)

        # 차원 맞추기
        landmarks = landmarks[:1629]
        if len(landmarks) < 1629:
            landmarks += [0.0] * (1629 - len(landmarks))

        return np.array(landmarks, dtype=np.float32)

    def _smooth_predictions(self, current_pred):
        """예측 평활화"""
        self.pred_history.append(current_pred)

        pred_counts = defaultdict(int)
        for pred in self.pred_history:
            pred_counts[pred] += 1

        return max(pred_counts.items(), key=lambda x: x[1])[0]

    def _process_sequence(self):
        if len(self.seq_buffer) < self.seq_length // 2:
            return

        sequence = np.stack(self.seq_buffer)
        sequence = sequence[:, self.topk_indices]
        mean = np.mean(sequence, axis=1, keepdims=True)
        std = np.std(sequence, axis=1, keepdims=True) + 1e-6
        sequence = (sequence - mean) / std

        inputs = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            confidence, pred = conf.item(), pred.item()

        if confidence > 0.3:
            self.confirmed_label = (pred, confidence)
            self.freeze_time = time.time()

    def _draw_results(self, frame):
        """프레임에 결과 시각화"""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 현재 인식 결과 표시
        if self.confirmed_label and (time.time() - self.freeze_time < 3):
            label, confidence = self.confirmed_label
            text = f"{self.index2label[label]} ({confidence:.2f})"
            draw.text((30, 30), text, font=self.font, fill=(0, 0, 0))

            color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0) if confidence > 0.3 else (255, 0, 0)
            draw.text((30, 30), text, font=self.font, fill=color)

        # 버퍼 상태 표시
        buffer_status = len(self.seq_buffer) / self.seq_length
        buffer_text = f"Buffer: {len(self.seq_buffer)}/{self.seq_length}"
        draw.text((30, 70), buffer_text, font=self.font, fill=(255, 255, 255))

        # 진행 바
        bar_width = 200
        bar_height = 10
        fill_width = int(bar_width * buffer_status)
        draw.rectangle([(30, 110), (30 + fill_width, 110 + bar_height)], fill=(0, 255, 0))
        draw.rectangle([(30, 110), (30 + bar_width, 110 + bar_height)], outline=(255, 255, 255), width=1)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def run(self):
        """메인 루프(전체화면+키포인트 시각화)"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("오류: 카메라를 열 수 없습니다")

        print("프로그램 시작, ESC키로 종료")
        print("Tip: 손이 카메라에 잘 보이게 해주세요")

        # OpenCV 전체화면 설정
        window_name = "Sign Language Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("경고: 프레임 읽기 실패")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.holistic.process(rgb_frame)

                # 키포인트 시각화
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)

                landmarks = self._extract_landmarks(results)

                if np.any(landmarks):
                    self.seq_buffer.append(landmarks)

                if len(self.seq_buffer) >= self.seq_length // 3:
                    prediction = self._process_sequence()
                    if prediction:
                        self.confirmed_label = prediction

                frame = self._draw_results(frame)
                cv2.imshow(window_name, frame)

                if cv2.waitKey(10) & 0xFF == 27:  # ESC
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            print("프로그램 종료")

if __name__ == "__main__":
    # 파일 존재 여부 확인
    REQUIRED_FILES = {
        "모델 파일": "best_model.pth",
        "라벨 파일": "data/KETI_top100_clean_words.xlsx",
        "특성 인덱스": "topk_indices.pth"
    }

    missing = [name for name, path in REQUIRED_FILES.items() if not os.path.exists(path)]
    if missing:
        print("오류: 필수 파일이 없습니다!")
        for name in missing:
            print(f"- {name}: {REQUIRED_FILES[name]}")
        exit(1)

    try:
        recognizer = SignLanguageRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"프로그램 오류: {str(e)}")
        exit(1)
