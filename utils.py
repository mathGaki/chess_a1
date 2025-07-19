import torch
import numpy as np
from collections import deque
import threading
from typing import Dict, Optional, Tuple

# core 폴더의 모듈 임포트
# 이 파일이 src/ 폴더에 위치하므로, 경로 설정이 필요할 수 있습니다.
from ..core.constants import white, black, pawn, knight, bishop, rook, queen, king, wk, wq, bk, bq
from ..core.moves import get_move_source, get_move_target, get_move_piece, get_move_promote_to
from ..core.bb_operations import get_ls1b_index
from ..core.position import Position
from .config import INPUT_CHANNELS, INPUT_HISTORY_STEPS, INPUT_CHANNELS_PER_STEP, POLICY_PLANES, QUEEN_MOVE_PLANES, KNIGHT_MOVE_PLANES

# 퀸/나이트 이동 방향 벡터 (dy, dx) - (rank, file) 변화량
# N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_DIRECTIONS = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]

# 언더프로모션 방향 벡터 (dy, dx)
# 플레이어 관점 (폰이 앞으로 가는 것을 -1로 간주)
PROMOTION_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1)] # 대각선 왼쪽, 전진, 대각선 오른쪽

# --- core.moves (placeholder functions) ---
# move는 일반적으로 시작 스퀘어, 타겟 스퀘어, 기물 타입, 프로모션 타입 등을 인코딩한 정수입니다.
# 실제 체스 엔진의 move 인코딩 방식에 따라 구현되어야 합니다.
# 실제 core.moves의 인코딩과 일치하도록 수정
def get_move_source(move): return move & 0x3f  # 6비트 (0-63)
def get_move_target(move): return (move & 0xfc0) >> 6  # 6비트 (0-63)
def get_move_piece(move): return (move & 0x7000) >> 12  # 3비트 (0-5 기물 타입)
def get_move_promote_to(move): return (move & 0xf0000) >> 16  # 4비트 (0-15 프로모션 기물 타입)

def make_move(position: Position, move: int) -> Position:
    """
    주어진 position에 move를 적용하여 새로운 position을 반환합니다.
    core.moves의 make_move 함수를 사용합니다.
    """
    from ..core.moves import make_move as core_make_move
    
    try:
        result = core_make_move(position, move)
        if result is None:
            # 디버깅을 위한 move 정보 출력 (필요시 제거 가능)
            source = get_move_source(move)
            target = get_move_target(move) 
            piece = get_move_piece(move)
            print(f"Warning: make_move returned None for move {move} (source={source}, target={target}, piece={piece})")
        return result
    except Exception as e:
        print(f"Error in make_move: {e} for move {move}")
        return None

def bb_to_plane(bb: np.uint64) -> np.ndarray:
    """64비트 비트보드를 8x8 numpy 배열로 변환"""
    s = np.binary_repr(bb, width=64)
    # FEN과 동일하게 8행부터 1행까지 (상단부터 하단) 순서로 변환
    # 비트보드는 일반적으로 LS1B가 a1 (0), MS1B가 h8 (63)
    # 8x8 배열로 reshape 후 상하 반전 (행 순서 뒤집기)
    return np.array(list(s), dtype=np.float32).reshape(8, 8)[::-1, :]

def _move_to_policy_index_impl(move: int, side: int) -> int:
    """
    core.moves의 정수 move를 4672 크기의 정책 벡터 인덱스로 변환하는 실제 로직.
    이 함수는 AlphaZero 논문 (Methods -> Representation)의 체스 액션 인코딩을 따릅니다.
    """
    source = get_move_source(move)
    target = get_move_target(move)
    piece_type = get_move_piece(move)
    promoted_to = get_move_promote_to(move)
    
    source_rank, source_file = source // 8, source % 8
    target_rank, target_file = target // 8, target % 8
    
    dy, dx = target_rank - source_rank, target_file - source_file
    
    # 플레이어 관점으로 변환 (흑의 경우 방향 반전)
    if side == black:
        dy, dx = -dy, -dx

    plane_idx = -1

    # 1. 언더프로모션 처리 (나이트, 비숍, 룩으로의 프로모션)
    # 폰이 7랭크에서 8랭크로 이동할 때 발생
    if promoted_to in (knight, bishop, rook):
        # 프로모션은 항상 폰의 전진 또는 대각선 전진 이동과 결합됩니다.
        if dx == -1: direction_idx = 0 # 대각선 왼쪽
        elif dx == 0: direction_idx = 1 # 전진
        elif dx == 1: direction_idx = 2 # 대각선 오른쪽
        else: return -1 # 유효하지 않은 프로모션 방향

        if promoted_to == knight: promote_offset = 0
        elif promoted_to == bishop: promote_offset = 1
        elif promoted_to == rook: promote_offset = 2
        else: return -1 # 유효하지 않은 프로मो션 타입 (퀸은 일반 퀸 이동으로 처리)
        
        plane_idx = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + (direction_idx * 3 + promote_offset)

    # 2. 나이트 이동 처리
    elif piece_type == knight:
        try:
            plane_idx = QUEEN_MOVE_PLANES + KNIGHT_DIRECTIONS.index((dy, dx))
        except ValueError:
            return -1 # 유효하지 않은 나이트 이동 방향

    # 3. 퀸 이동 처리 (폰, 비숍, 룩, 퀸, 킹의 일반 이동 및 퀸으로의 프로모션)
    # 폰의 일반 이동, 킹의 이동, 퀸/룩/비숍의 모든 이동을 포함
    else:
        dy_abs, dx_abs = abs(dy), abs(dx)
        # 직선 또는 대각선 이동이 아니면 유효하지 않음
        if not (dy == 0 or dx == 0 or dy_abs == dx_abs): return -1
        
        # 이동 거리 (1부터 7까지)
        distance = max(dy_abs, dx_abs)
        if not (1 <= distance <= 7): return -1

        # 방향 벡터 정규화
        direction_dy = dy // distance if distance != 0 else 0
        direction_dx = dx // distance if distance != 0 else 0
        
        try:
            direction_idx = QUEEN_DIRECTIONS.index((direction_dy, direction_dx))
            plane_idx = direction_idx * 7 + (distance - 1) # 8방향 * 7거리
        except ValueError:
            return -1 # 유효하지 않은 퀸 이동 방향

    if plane_idx == -1: return -1
    
    # 출발지 스퀘어 인덱스 (0-63)
    # 흑의 경우 플레이어 관점으로 보드를 뒤집어야 하므로 출발지 랭크를 변환
    if side == black:
        source_square_idx = (7 - source_rank) * 8 + source_file
    else:
        source_square_idx = source_rank * 8 + source_file

    # 최종 정책 인덱스 = (출발지 스퀘어 * POLICY_PLANES) + 평면 인덱스
    return source_square_idx * POLICY_PLANES + plane_idx

def move_to_policy_index(move: int, side: int) -> int:
    """
    게임의 'move' 객체를 정책 벡터의 인덱스 (0부터 ACTION_SPACE_SIZE-1)로 변환합니다.
    _move_to_policy_index_impl을 호출합니다.
    """
    return _move_to_policy_index_impl(move, side)

def policy_index_to_move(index: int) -> int:
    """
    정책 벡터의 인덱스를 게임의 'move' 객체로 변환합니다.
    move_to_policy_index의 역함수입니다.
    이 함수는 구현하기 매우 복잡하며, 실제 게임 엔진의 move 생성 로직이 필요합니다.
    """
    # 플레이스홀더: 실제 move 객체로 변환 필요
    # 역변환을 위해서는 index로부터 source_square, plane_idx를 추출하고
    # 이를 다시 dy, dx, distance, promoted_to 등으로 변환하여 move 객체로 재구성해야 합니다.
    return index 

def position_to_tensor(pos: Position, history_buffer: deque = None) -> torch.Tensor:
    """
    Position 객체와 히스토리 버퍼를 받아 신경망 입력 텐서로 변환.
    history_buffer는 Position 객체의 deque(maxlen=INPUT_HISTORY_STEPS)여야 합니다.
    AlphaZero 논문 (Methods -> Representation)에 따라 입력은 N x N x (MT + L) 형태의
    이미지 스택으로, 보드 상태, 턴 정보, 특수 규칙 등을 여러 평면으로 인코딩합니다.
    체스의 경우 8x8x119 텐서가 될 수 있습니다.
    """
    HEIGHT = 8 # 체스 보드 높이
    WIDTH = 8  # 체스 보드 너비

    # 1. 기물 위치 히스토리 채널 (T * M)
    piece_planes = []
    # 현재부터 과거 순으로 히스토리 순회
    # history_buffer가 제공되지 않으면 현재 pos만 사용
    history_list = list(history_buffer) if history_buffer is not None else [pos]

    for old_pos in history_list:
        # None 체크 추가
        if old_pos is None:
            print("Warning: None position encountered in position_to_tensor")
            continue
            
        # 플레이어 관점으로 보드 변환 (현재 플레이어가 항상 백)
        # old_pos.side는 해당 시점의 턴 플레이어
        p_color = old_pos.side
        o_color = 1 - p_color
        
        for piece_type in range(6): # 폰, 나이트, 비숍, 룩, 퀸, 킹
            piece_planes.append(bb_to_plane(old_pos.pieces[p_color][piece_type]))
        for piece_type in range(6):
            piece_planes.append(bb_to_plane(old_pos.pieces[o_color][piece_type]))

    # 히스토리가 부족할 경우 0으로 채움
    while len(piece_planes) < INPUT_HISTORY_STEPS * INPUT_CHANNELS_PER_STEP:
        piece_planes.append(np.zeros((HEIGHT, WIDTH), dtype=np.float32))

    # 2. 상수 채널 (L)
    const_planes = []
    # 플레이어 색상 (항상 1, 왜냐하면 플레이어 관점이므로)
    const_planes.append(np.ones((HEIGHT, WIDTH), dtype=np.float32))
    
    # 총 이동 수 (정규화)
    # AlphaZero 논문 Table S1: Total move count 1 (plane)
    total_moves_normalized = pos.fullmove_number / 200.0 # 임의의 정규화 값 (예: 최대 200수 게임)
    const_planes.append(np.full((HEIGHT, WIDTH), total_moves_normalized, dtype=np.float32))
    
    # 캐슬링 권리 (P1 castling 2, P2 castling 2)
    # wk, wq, bk, bq는 core.constants에서 와야 함
    my_castle_wk = wk if pos.side == white else bk
    my_castle_wq = wq if pos.side == white else bq
    opp_castle_bk = bk if pos.side == white else wk
    opp_castle_bq = bq if pos.side == white else wq

    const_planes.append(np.ones((HEIGHT, WIDTH), dtype=np.float32) if (pos.castle & my_castle_wk) else np.zeros((HEIGHT, WIDTH), dtype=np.float32))
    const_planes.append(np.ones((HEIGHT, WIDTH), dtype=np.float32) if (pos.castle & my_castle_wq) else np.zeros((HEIGHT, WIDTH), dtype=np.float32))
    const_planes.append(np.ones((HEIGHT, WIDTH), dtype=np.float32) if (pos.castle & opp_castle_bk) else np.zeros((HEIGHT, WIDTH), dtype=np.float32))
    const_planes.append(np.ones((HEIGHT, WIDTH), dtype=np.float32) if (pos.castle & opp_castle_bq) else np.zeros((HEIGHT, WIDTH), dtype=np.float32))
    
    # 50수 규칙 카운터 (No-progress count 1)
    no_progress_count_normalized = pos.halfmove_clock / 50.0 # 50수 규칙까지
    const_planes.append(np.full((HEIGHT, WIDTH), no_progress_count_normalized, dtype=np.float32))

    # AlphaZero 논문 Table S1의 "Repetitions 2"는 현재 pos에 직접적인 필드가 없으므로 0으로 채움
    # 실제 구현 시에는 Position 클래스에 반복 횟수 필드를 추가하고 업데이트해야 합니다.
    const_planes.append(np.zeros((HEIGHT, WIDTH), dtype=np.float32)) # Repetitions plane 1
    const_planes.append(np.zeros((HEIGHT, WIDTH), dtype=np.float32)) # Repetitions plane 2


    # 모든 채널을 합쳐 텐서로 변환
    all_planes = np.stack(piece_planes + const_planes, axis=0)
    
    # 최종 텐서의 채널 수가 INPUT_CHANNELS와 일치하는지 확인 (디버깅용)
    if all_planes.shape[0] != INPUT_CHANNELS:
        print(f"Warning: Generated tensor channels ({all_planes.shape[0]}) do not match INPUT_CHANNELS ({INPUT_CHANNELS}).")
        # 필요에 따라 채널을 자르거나 패딩할 수 있습니다.
        if all_planes.shape[0] > INPUT_CHANNELS:
            all_planes = all_planes[:INPUT_CHANNELS, :, :]
        elif all_planes.shape[0] < INPUT_CHANNELS:
            padding = np.zeros((INPUT_CHANNELS - all_planes.shape[0], HEIGHT, WIDTH), dtype=np.float32)
            all_planes = np.concatenate((all_planes, padding), axis=0)

    return torch.from_numpy(all_planes)


class TensorCache:
    """Position-to-tensor conversion cache for performance optimization"""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[int, torch.Tensor] = {}
        self._access_order: deque = deque()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, pos: Position, position_history: Optional[deque] = None) -> torch.Tensor:
        """Get cached tensor or compute and cache new one"""
        cache_key = self._get_cache_key(pos)
        
        with self._lock:
            if cache_key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return self._cache[cache_key].clone()  # Return copy to avoid modifications
            
            self._misses += 1
            # Compute new tensor
            tensor = position_to_tensor(pos, position_history)
            
            # Add to cache
            self._add_to_cache(cache_key, tensor)
            return tensor.clone()
    
    def _get_cache_key(self, pos: Position) -> int:
        """Generate cache key from position hash"""
        # Use position hash if available, otherwise compute simple hash
        if hasattr(pos, 'hash_key') and pos.hash_key != 0:
            return pos.hash_key
        
        # Simple hash based on piece positions
        hash_val = 0
        for color in range(2):
            for piece_type in range(6):
                hash_val ^= pos.pieces[color][piece_type] * (color * 6 + piece_type + 1)
        
        hash_val ^= pos.side << 40
        hash_val ^= pos.enpas << 45
        hash_val ^= pos.castle << 50
        
        return hash_val & 0x7FFFFFFFFFFFFFFF  # Ensure positive
    
    def _add_to_cache(self, key: int, tensor: torch.Tensor):
        """Add tensor to cache with LRU eviction"""
        if len(self._cache) >= self._max_size:
            # Remove least recently used
            lru_key = self._access_order.popleft()
            del self._cache[lru_key]
        
        self._cache[key] = tensor.clone()
        self._access_order.append(key)
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache),
                'max_size': self._max_size
            }

# Global tensor cache
_tensor_cache = TensorCache()

def position_to_tensor_cached(pos: Position, position_history: Optional[deque] = None) -> torch.Tensor:
    """Cached version of position_to_tensor"""
    return _tensor_cache.get(pos, position_history)

def clear_tensor_cache():
    """Clear the global tensor cache"""
    _tensor_cache.clear()

def get_tensor_cache_stats() -> Dict[str, int]:
    """Get tensor cache statistics"""
    return _tensor_cache.get_stats()

