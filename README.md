# AI 폴더 구조

## 핵심 파일들 (Production)
- `config.py` - 모든 설정 파라미터 (32 워커 설정)
- `neural_network.py` - 체스 신경망 모델 
- `mcts.py` - 몬테카를로 트리 탐색 구현
- `stable_inference.py` - 안정적인 추론 서버 (메인 시스템)
- `game_player.py` - 공통 게임 플레이 로직 (셀프플레이/평가)
- `selfplay.py` - 셀프플레이 유틸리티 함수들
- `train.py` - 신경망 훈련 시스템
- `evaluate.py` - 모델 평가 시스템
- `gui.py` - GUI 인터페이스 (32개 보드 8x4 배치)
- `utils.py` - 유틸리티 함수들
- `object_pool.py` - 메모리 풀 관리
- `warmup.py` - numba 워밍업
- `inference_server.py` - 기존 추론 서버 (백업)

## 실험용 파일들 (experimental/)
- `shared_memory_inference.py` - 공유 메모리 추론 시스템
- `process_safe_shared_memory.py` - 프로세스 안전 공유 메모리
- `ultra_optimized_inference.py` - 최적화된 추론 시스템
- `true_shared_memory.py` - 실제 공유 메모리 구현
- `threaded_inference.py` - 멀티스레드 추론
- `shared_memory_worker.py` - 공유 메모리 워커
- `evaluate_new.py` - 새로운 평가 시스템

## Import 관계
```
main.py
├── stable_inference.py
│   ├── game_player.py
│   │   ├── selfplay.py (유틸리티 함수)
│   │   └── mcts.py
│   └── neural_network.py
├── train.py
└── evaluate.py
    └── game_player.py
```

## 사용 방법
1. 메인 훈련: `python -m src.main`
2. 개별 테스트: 각 파일의 `if __name__ == "__main__":` 블록 사용

## GUI 설정 (32 워커 지원)
- 32개 체스판을 8x4 배치로 표시
- 각 워커가 고유한 보드에 게임 상태 표시
- 창 크기: 1800x1000 (32개 보드 최적화)
- 보드 크기: 200x200 픽셀 (32개 보드용 축소)
