import torch
import torch.nn as nn

class AudioEmotionNet(nn.Module):
    def __init__(self, input_dim=40, num_classes=8, cnn_out=128, lstm_hidden=64):
        super().__init__()
        
        # CNN部分：提取局部特征
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 时间维减半
        )
        
        # BiLSTM：建模时间序列
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),  # 双向LSTM，输出维度是2倍隐藏层
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # → (B, F, T)
        x = self.feature_extractor(x)  # → (B, C=cnn_out, T')
        x = x.permute(0, 2, 1)  # → (B, T', C)
        
        lstm_out, _ = self.lstm(x)  # → (B, T', 2*lstm_hidden)
        x = torch.mean(lstm_out, dim=1)  # → (B, 2*lstm_hidden)
        
        return self.classifier(x)  # → (B, num_classes)