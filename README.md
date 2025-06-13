#실시간 수어 인식 기반 텍스트 출력 시스템

게임콘텐츠 캡스톤디자인(SWCON36700) 25-1

한채연(팀장), 황해연, 최성우, LI HAISONG

가상 환경 세팅  
sign_env.yml  
conda env create -f sign_env.yml  
conda activate sign_env

데이텃셋  
KETI수어:https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=264
WLASL수어:https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized

열할분담  
한채연 - LSTM + GRU  
황해연 - LSTM  
최성우 - GRU  
LI HAISONG - Transformer

구조
- dataset/        # 전처리코드 및 처리된 xlsx
- GRU/            # GRU 모델 학습 코드
- LSTM/           # LSTM 모델 학습 코드
- LSTM+GRU/       # LSTM + GRU를 학습 및 실시간 예측 코드
- Transformer/    # Transformer 학습 및 실시간 예측 코드
- sign_env.yml    # Conda 환경 설정 파일
- README.txt      # 프로젝트 문서화 파일
