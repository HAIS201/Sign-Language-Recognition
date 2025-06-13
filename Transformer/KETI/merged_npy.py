import os
import numpy as np
from tqdm import tqdm  # 진행률 표시

def merge_npy_sequence(folder_path, save_name="merged.npy"):
    npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy") and f != save_name]
    # 병합 대상 프레임 파일들 목록
    if not npy_files:
        return False  # .npy 파일이 없으면 종료

    # 프레임 번호 기준으로 정렬
    npy_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    frames = []
    for fname in npy_files:
        path = os.path.join(folder_path, fname)
        data = np.load(path)  # 프레임 불러오기
        if data.shape != (1629,):  # 키포인트 차원 체크
            continue
        frames.append(data)  # 유효한 프레임 추가

    if len(frames) == 0:
        return False  # 병합할 프레임이 없으면 종료

    sequence = np.stack(frames)  # (T, 1629) 시퀀스로 스택
    np.save(os.path.join(folder_path, save_name), sequence)  # 하나의 파일로 저장
    return True  # 병합 성공

def process_dataset(root_dir, include_aug=True):
    print(f"목록: {root_dir}")  # 처리 시작 알림
    total = 0  # 총 폴더 수
    success = 0  # 병합 성공 수

    for label_name in tqdm(os.listdir(root_dir)):  # 레이블 단위 반복
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        for video_folder in os.listdir(label_path):  # 각 비디오 폴더 반복
            if not include_aug and "_aug" in video_folder:
                continue  # 증강 폴더 스킵

            folder_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(folder_path):
                continue

            total += 1
            ok = merge_npy_sequence(folder_path)  # 병합 시도
            if ok:
                success += 1

    print(f"완료：{success}/{total} 저장: merged.npy")  # 결과 출력

if __name__ == "__main__":
    process_dataset(r"D:\pycharm\transformer\sign_language_transformer\data\MP_Data", include_aug=True)
    process_dataset(r"D:\pycharm\transformer\sign_language_transformer\data\MP_Data_Test", include_aug=True)
