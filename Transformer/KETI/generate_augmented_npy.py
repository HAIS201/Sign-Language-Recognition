import os
import numpy as np
from tqdm import tqdm
import shutil

# 노이즈 추가 함수
def add_noise(data, std=0.01):
    noise = np.random.normal(0, std, data.shape)  # 표준편차 std에 따라 노이즈 생성
    return data + noise  # 원본에 노이즈 더하기

# 좌우 반전 함수
def flip_horizontal(data):
    points = data.reshape(-1, 3)  # (N, 3) 형태로 변형
    points[:, 0] *= -1  # X 좌표 반전
    return points.reshape(data.shape)  # 원래 형태로 복원

# 평면 회전 함수
def rotate_xy(data, angle_deg):
    angle = np.deg2rad(angle_deg)  # 각도를 라디안으로 변환
    points = data.reshape(-1, 3)
    xy = points[:, :2]
    z = points[:, 2:]
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    xy_rotated = xy @ rotation_matrix.T
    return np.hstack([xy_rotated, z]).reshape(data.shape)

# 시간 마스킹 함수
def apply_masking(data, mask_ratio=0.2):
    num_dims = data.shape[0]
    mask_num = int(num_dims * mask_ratio)
    mask_indices = np.random.choice(num_dims, mask_num, replace=False)
    data[mask_indices] = 0
    return data

# 변환 적용 및 저장 함수
def process_and_save(src_folder, dst_folder, transform_fn):
    os.makedirs(dst_folder, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.npy')],
                         key=lambda x: int(x.split('.')[0]))  # 프레임 번호 순 정렬
    for file in frame_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        data = np.load(src_path)
        transformed = transform_fn(data.copy())
        np.save(dst_path, transformed)

# 전체 데이터셋에 증강 적용
def generate_augmented_dataset(root_dir):
    for label_name in tqdm(os.listdir(root_dir), desc="Processing labels"):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        for video_folder in os.listdir(label_path):
            if "_aug" in video_folder:
                continue
            src_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(src_path):
                continue

            aug_variants = {
                "_aug_noise": lambda x: add_noise(x, std=0.01),
                "_aug_flip": flip_horizontal,
                "_aug_rot": lambda x: rotate_xy(x, angle_deg=10),
                "_aug_mask": lambda x: apply_masking(x, mask_ratio=0.2)
            }

            for suffix, func in aug_variants.items():
                dst_path = src_path + suffix
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                process_and_save(src_path, dst_path, func)

# 메인 함수 실행
if __name__ == "__main__":
    base_dir = r"D:\pycharm\transformer\sign_language_transformer\data\MP_Data"
    generate_augmented_dataset(base_dir)
    print("All augmentations done.")
