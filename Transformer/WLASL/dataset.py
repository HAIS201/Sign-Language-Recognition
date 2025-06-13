# dataset.py - 수화 동작 프레임 시퀀스 데이터셋 정의 / 手语动作帧序列数据集定义（支持标签映射、增强与归一化）
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import normalize_frames

# 프레임 리스트를 일정 길이로 리샘플링 / 将帧列表重采样为固定长度
def sample_paths(frame_list, target_len=60):
    """
    프레임 시퀀스를 선형 보간 또는 균등 샘플링하여 일정 길이로 맞춤 / 对帧列表均匀采样或线性插值，返回定长序列
    """
    total_len = len(frame_list)
    if total_len == 0:
        raise ValueError("Empty frame list")
    frame_array = np.array(frame_list)
    if total_len >= target_len:
        idxs = np.linspace(0, total_len - 1, target_len).astype(int)
        return [frame_array[i] for i in idxs]
    else:
        new_idxs = np.linspace(0, total_len - 1, target_len)
        interpolated = []
        for i in new_idxs:
            low = int(np.floor(i))
            high = int(np.ceil(i))
            alpha = i - low
            if high >= total_len:
                high = total_len - 1
            interpolated_frame = (1 - alpha) * frame_array[low] + alpha * frame_array[high]
            interpolated.append(interpolated_frame)
        return interpolated

# 수화 데이터셋 클래스 정의 / 手语帧序列数据集类定义
class SignDataset(Dataset):
    """
    수화 프레임 시퀀스를 로드하는 데이터셋 클래스 / 用于加载手语帧序列的数据集类
    root_dir: MP_Data_Train 또는 MP_Data_Test 디렉토리 / 数据根目录
    label_file: Excel 파일 (파일명, 어휘명 포함) / 包含文件名与词汇名的Excel
    label2index: 라벨 인코딩 매핑 (옵션) / 标签到索引的映射（可选）
    """
    def __init__(self, root_dir, label_file, label2index=None, frame_len=60, train=False):
        self.train = train
        self.root_dir = root_dir
        self.label_df = pd.read_excel(label_file)
        self.frame_len = frame_len
        self.samples = []

        # 라벨 파일 내 모든 데이터 탐색 / 遍历标签文件中的所有数据
        total_rows = len(self.label_df)
        print(f"총 {total_rows}개의 라벨을 탐색합니다...")

        for idx, (_, row) in enumerate(self.label_df.iterrows()):
            folder_name = str(row["파일명"]).split(".")[0]
            label_name = str(row["어휘명"]).strip()
            base_folder_path = os.path.join(self.root_dir, label_name, folder_name)

            all_paths = [base_folder_path]
            parent_dir = os.path.dirname(base_folder_path)
            base_name = os.path.basename(base_folder_path)

            # 증강된 데이터 서브폴더 자동 수집 / 自动收集所有增强子目录
            if os.path.exists(parent_dir):
                for subfolder in os.listdir(parent_dir):
                    if subfolder.startswith(base_name + "_aug"):
                        all_paths.append(os.path.join(parent_dir, subfolder))

            for folder_path in all_paths:
                if os.path.exists(folder_path):
                    self.samples.append((folder_path, label_name))

            if idx % 500 == 0:
                print(f"진행 중: {idx}/{total_rows} 라벨 처리 완료...")

        print(f"총 {len(self.samples)}개의 샘플 로딩 완료!")

        # 라벨 인덱싱 및 매핑 / 标签编码与映射
        all_labels = sorted(set([label for _, label in self.samples]))
        self.label2index = {label: idx for idx, label in enumerate(all_labels)}
        self.index2label = {idx: label for label, idx in self.label2index.items()}
        self.label_map = self.label2index

        # 외부 라벨 매핑이 주어졌을 때 덮어쓰기 / 如果提供了外部标签映射，则覆盖
        if label2index is not None:
            self.label2index = label2index
            self.index2label = {idx: label for label, idx in label2index.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label_name = self.samples[idx]
        merged_path = os.path.join(folder_path, "merged.npy")

        # 병합된 npy 파일이 존재할 경우 우선 사용 / 优先使用 merged.npy 加速读取
        if os.path.exists(merged_path):
            sequence = np.load(merged_path)
            if sequence.ndim != 2 or sequence.shape[1] != 1629:
                raise ValueError(f"{merged_path} 파일 형식 오류, shape: {sequence.shape}")
            if sequence.shape[0] != self.frame_len:
                sequence = sample_paths(list(sequence), target_len=self.frame_len)
        else:
            # 프레임별 npy 파일 직접 로드 / 若无 merged.npy 则逐帧读取 .npy 文件
            npy_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith(".npy") and f.split('.')[0].isdigit()
            ], key=lambda x: int(x.split('.')[0]))
            if len(npy_files) == 0:
                raise ValueError(f"{folder_path} 폴더에 npy 프레임이 없습니다.")

            frames = []
            for filename in npy_files:
                full_path = os.path.join(folder_path, filename)

                data = np.load(full_path)

                # (N, 3) 형태의 좌표 정보를 플랫하게 변환
                if data.ndim == 2 and data.size == 1629:
                    data = data.reshape(1629)

                # 여전히 조건 불만족이면 건너뜀
                if data.shape != (1629,):
                    print(f"무시됨: {full_path}, shape: {data.shape}")
                    continue

                frames.append(data)

            #  추가: 빈 프레임 예외 처리
            if len(frames) == 0:
                print(f"경고: {folder_path} 안에 유효한 프레임이 없음. 샘플 건너뜀.")
                return self.__getitem__((idx + 1) % len(self))

            frames = sample_paths(frames, target_len=self.frame_len)
            sequence = np.stack(frames)

        # 프레임 정규화 수행 / 对帧进行归一化处理
        sequence = normalize_frames(sequence)
        sequence = torch.tensor(np.array(sequence), dtype=torch.float32)

        label = torch.tensor(self.label2index[label_name], dtype=torch.long)
        return sequence, label
