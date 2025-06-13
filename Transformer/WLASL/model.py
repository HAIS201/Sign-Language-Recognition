import torch
import torch.nn as nn
import math

# 위치 인코딩 클래스 / 位置编码模块（用于给 Transformer 加时间顺序信息）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원은 sin / 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원은 cos / 奇数维度使用 cos

        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 로 확장 / 扩展 batch 차원
        self.register_buffer("pe", pe)  # 학습은 하지 않지만 저장되는 버퍼로 등록 / 注册为 buffer，不参与训练但会保存

    def forward(self, x):
        """
        x 입력: [batch_size, seq_len, d_model]
        출력: 위치 인코딩이 더해진 x / 输出：加上位置编码的张量
        """
        return x + self.pe[:, :x.size(1), :]

# Transformer 기반 수어 분류 모델 정의 / 定义基于 Transformer 的手语识别模型
class SignTransformer(nn.Module):
    def __init__(self, input_dim=1629, d_model=512, nhead=8, num_layers=4, num_classes=50, dropout=0.3):
        super().__init__()

        # 입력 차원 -> d_model 선형 변환 / 将输入维度映射到 d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 드롭아웃 / Dropout 层用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

        # 위치 인코딩 / 位置编码模块
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 인코더 구성 / 构建 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True  # 입력 shape 을 [B, T, D] 로 사용 / 使用 [B, T, D] 格式输入
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers  # 인코더 레이어 수 / 编码器层数
        )

        # Attention Pooling 을 위한 선형 레이어 / 注意力池化用线性层
        self.attn_fc = nn.Linear(d_model, 1)

        # 최종 분류기 정의 / 定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        입력 x: [batch_size, seq_len, input_dim]
        출력: [batch_size, num_classes]
        """
        x = self.input_proj(x)       # 입력 특징 차원 조정 / 特征维度映射
        x = self.dropout(x)          # 드롭아웃 적용 / 应用 Dropout
        x = self.pos_encoder(x)      # 위치 인코딩 추가 / 添加位置编码
        x = self.transformer_encoder(x)  # Transformer 인코더 통과 / 通过 Transformer 编码器

        # Attention Pooling 수행 / 执行注意力池化
        attn_weights = torch.softmax(self.attn_fc(x), dim=1)  # attention 가중치 계산 / 计算注意力权重
        x = torch.sum(attn_weights * x, dim=1)                # 가중 합 / 加权求和

        out = self.classifier(x)     # 최종 예측 / 最终分类输出
        return out
