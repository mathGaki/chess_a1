"""
numba 함수들을 미리 워밍업하는 유틸리티
"""
import time
from ..core.position import parse_fen
from ..core.moves import generate_legal_moves_fast
from ..core.constants import start_position

def warmup_numba_functions():
    """
    애플리케이션 시작시 numba 함수들을 미리 컴파일
    """
    print("🔥 Warming up numba functions...")
    start_time = time.time()
    
    # 기본 체스 위치로 워밍업
    pos = parse_fen(start_position)
    
    # 핵심 함수들 워밍업 (여러 번 호출하여 확실히 컴파일)
    for _ in range(10):
        _ = generate_legal_moves_fast(pos)
    
    # 추가 위치에서도 워밍업 (다양한 패턴 컴파일)
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3"
    ]
    
    for test_fen in test_positions:
        test_pos = parse_fen(test_fen)
        for _ in range(3):
            _ = generate_legal_moves_fast(test_pos)
    
    end_time = time.time()
    print(f"✅ numba warmup completed in {end_time - start_time:.2f}s")
    print("🚀 All functions are now compiled and ready for high-speed execution")

if __name__ == "__main__":
    warmup_numba_functions()
