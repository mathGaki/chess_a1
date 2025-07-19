# src/neural_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    INPUT_CHANNELS, ACTION_SPACE_SIZE,
    NN_NUM_CHANNELS, NN_NUM_RES_BLOCKS, 
    NN_POLICY_HEAD_CHANNELS, NN_VALUE_HEAD_CHANNELS, NN_VALUE_HIDDEN_SIZE
)

# AlphaZero 신경망 아키텍처 (256 filters, 19 residual blocks)

class ResidualBlock(nn.Module):
    """ResNet의 기본 잔차 블록"""
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    """AlphaZero 스타일 체스 신경망 (256 filters, 19 residual blocks)"""
    def __init__(self, num_channels=None, num_res_blocks=None):
        super(ChessNet, self).__init__()
        
        # AlphaZero 기본 설정 사용
        if num_channels is None:
            num_channels = NN_NUM_CHANNELS
        if num_res_blocks is None:
            num_res_blocks = NN_NUM_RES_BLOCKS
            
        # 공통 Body
        self.conv_in = nn.Conv2d(INPUT_CHANNELS, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # 정책 헤드 (Policy Head) - AlphaZero 스펙
        self.policy_conv = nn.Conv2d(num_channels, NN_POLICY_HEAD_CHANNELS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(NN_POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(NN_POLICY_HEAD_CHANNELS * 8 * 8, ACTION_SPACE_SIZE)

        # 가치 헤드 (Value Head) - AlphaZero 스펙
        self.value_conv = nn.Conv2d(num_channels, NN_VALUE_HEAD_CHANNELS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(NN_VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(NN_VALUE_HEAD_CHANNELS * 8 * 8, NN_VALUE_HIDDEN_SIZE)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Body
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)
        # 확률 분포로 만들기 위해 LogSoftmax 사용 (학습 시 NLLLoss와 결합)
        policy_log_probs = F.log_softmax(policy, dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # -1과 1 사이의 값으로 출력

        return policy_log_probs, value