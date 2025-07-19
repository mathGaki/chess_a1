import torch

#MCTS - 5천만 판 최적화
NUM_ITERATIONS = 50000  # 5000 → 50000 (50M 게임을 위한 긴 훈련)
NUM_WORKERS = 64  # 32 → 64 (더 많은 병렬 처리로 배치 크기 증가)
GAMES_PER_ITERATION = 1000  # iteration당 게임 수
MCTS_SIMULATIONS = 100  # 200 → 100 (속도 최적화)

# MCTS 탐색 설정 (AlphaZero 논문 기준)
MCTS_C_PUCT = 1.25  # UCT 탐색 계수 (AlphaZero 논문값)
MCTS_DIRICHLET_ALPHA = 0.3  # 체스용 Dirichlet noise alpha
MCTS_DIRICHLET_WEIGHT = 0.25  # 노이즈 가중치 (AlphaZero 논문값: 25%)

# MCTS 온도 설정 (AlphaZero 논문 기준)
MCTS_TEMPERATURE_EARLY = 1.0  # 초기 게임 온도 (탐험적 플레이)
MCTS_TEMPERATURE_LATE = 0.0   # 후기 게임 온도 (결정적 플레이, AlphaZero 방식)
MCTS_TEMPERATURE_THRESHOLD = 30  # 온도 전환 기준 (수)

INFERENCE_BATCH_SIZE = 1600  # 기본 배치 크기 (32워커 × 100노드 = 3200 대비)
INFERENCE_TIMEOUT = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Architecture (AlphaZero 기준이지만 메모리 최적화)
NN_NUM_CHANNELS = 256  # Conv layer filters (256 → 128, 메모리 절약)
NN_NUM_RES_BLOCKS = 48  # Residual blocks (48 → 12, 메모리 절약)
NN_POLICY_HEAD_CHANNELS = 2  # Policy head conv channels
NN_VALUE_HEAD_CHANNELS = 1   # Value head conv channels
NN_VALUE_HIDDEN_SIZE = 256   # Value head FC layer size

# Train (AlphaZero 논문 기준 + RTX 5070 최적화)
learning_rate = 0.001  # 0.2 → 0.01 (NaN 방지를 위한 안전한 값)
weight_decay = 1e-4   # L2 정규화
batch_size = 256
training_steps = 1000  # AlphaZero: 각 iteration마다 700k steps (우리는 700 steps로 축소)

# Performance Optimization (RTX 5070 최적화)
ENABLE_TORCH_COMPILE = True   # Triton 설치 완료! 1.5-2배 속도 향상 기대
ENABLE_MIXED_PRECISION = True # Mixed Precision Training 활성화 (2배 속도 향상)
TORCH_COMPILE_MODE = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'

# AlphaZero Learning Rate Schedule (매우 안전한 값으로 조정)
LR_SCHEDULE = {
    0: 0.001,       # 0-100k steps (0.01 → 0.001, 더 안전한 값)
    100000: 0.0002, # 100k-300k steps 
    300000: 0.00005, # 300k-500k steps
    500000: 0.00001 # 500k+ steps
}

# Game Rules (성능 최적화)
INSUFFICIENT_MATERIAL_CHECK_INTERVAL = 10  # 기물부족 무승부 체크 간격 (수)

# Paths
model_dir = 'models'
data_dir = 'data/selfplay'
current_model = 'models/current_model.pth'
best_model = 'models/best_model.pth'

# Model Checkpointing (주기적 모델 저장)
CHECKPOINT_INTERVAL = 10  # 10 iteration마다 체크포인트 저장
CHECKPOINT_DIR = 'models/checkpoints'  # 체크포인트 저장 폴더

# Utils(Tensor)
INPUT_HISTORY_STEPS = 8
INPUT_CHANNELS_PER_STEP = 12 # 6 (my pieces) + 6 (opponent pieces)
INPUT_CONSTANT_CHANNELS = 9
INPUT_CHANNELS = INPUT_HISTORY_STEPS * INPUT_CHANNELS_PER_STEP + INPUT_CONSTANT_CHANNELS # 105

# 정책(Policy) 출력 (Representation 섹션 및 Table S2 기반)
# 8x8 출발지 * 73가지 이동 방식 = 4672가지 액션
POLICY_PLANES = 73
ACTION_SPACE_SIZE = 8 * 8 * POLICY_PLANES # 4672

# 73개 평면 상세 정의
QUEEN_MOVE_PLANES = 56
KNIGHT_MOVE_PLANES = 8
UNDERPROMOTION_PLANES = 9