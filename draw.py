"""
효율적인 무승부 조건 검사 모듈
"""

from numba import njit
import numpy as np
from .position import Position
from .moves import generate_legal_moves, is_in_check

@njit(cache=True)
def is_stalemate(pos):
    """
    스테일메이트 검사: 합법수가 없고 체크 상태가 아님
    
    Args:
        pos (Position): 현재 포지션
        
    Returns:
        bool: 스테일메이트 여부
    """
    # 체크 상태라면 스테일메이트가 아님
    if is_in_check(pos, pos.side):
        return False
    
    # 합법수가 없으면 스테일메이트
    legal_moves = generate_legal_moves(pos)
    return len(legal_moves) == 0

@njit(cache=True) 
def is_fifty_move_rule(pos):
    """
    50수 룰 검사: halfmove_clock이 100 이상 (50수 * 2턴)
    
    Args:
        pos (Position): 현재 포지션
        
    Returns:
        bool: 50수 룰 해당 여부
    """
    return pos.halfmove_clock >= 100

class DrawChecker:
    """무승부 조건 검사를 위한 클래스 (Zobrist 해시 히스토리 관리)"""
    
    def __init__(self):
        self.position_history = {}  # hash -> count
        self.move_history = []      # 되돌리기용
        
    def add_position(self, pos):
        """포지션을 히스토리에 추가"""
        hash_key = pos.hash_key
        self.position_history[hash_key] = self.position_history.get(hash_key, 0) + 1
        self.move_history.append(hash_key)
        
    def remove_last_position(self):
        """마지막 포지션을 히스토리에서 제거 (되돌리기용)"""
        if self.move_history:
            hash_key = self.move_history.pop()
            self.position_history[hash_key] -= 1
            if self.position_history[hash_key] == 0:
                del self.position_history[hash_key]
                
    def is_threefold_repetition(self, pos):
        """3수 동형 반복 검사"""
        hash_key = pos.hash_key
        return self.position_history.get(hash_key, 0) >= 2  # 현재 포지션까지 포함하면 3회
        
    def clear(self):
        """히스토리 초기화"""
        self.position_history.clear()
        self.move_history.clear()

def is_draw(pos, draw_checker=None):
    """
    통합 무승부 검사
    
    Args:
        pos (Position): 현재 포지션
        draw_checker (DrawChecker): 반복 검사용 (옵션)
        
    Returns:
        tuple: (is_draw, reason)
    """
    # 1. 스테일메이트 검사
    if is_stalemate(pos):
        return True, "Stalemate"
    
    # 2. 50수 룰 검사  
    if is_fifty_move_rule(pos):
        return True, "Fifty-move rule"
        
    # 3. 3수 동형 반복 검사 (draw_checker가 제공된 경우)
    if draw_checker and draw_checker.is_threefold_repetition(pos):
        return True, "Threefold repetition"
        
    return False, None

def is_insufficient_material(pos):
    """
    기물 부족 무승부 검사 (추가 최적화)
    - King vs King
    - King + Bishop vs King  
    - King + Knight vs King
    
    Args:
        pos (Position): 현재 포지션
        
    Returns:
        bool: 기물 부족 무승부 여부
    """
    # 각 색깔별 기물 개수 계산
    white_pieces = 0
    black_pieces = 0
    white_bishops = 0
    black_bishops = 0
    white_knights = 0  
    black_knights = 0
    
    for piece_type in range(6):  # pawn, knight, bishop, rook, queen, king
        white_count = bin(pos.pieces[0][piece_type]).count('1')
        black_count = bin(pos.pieces[1][piece_type]).count('1')
        
        if piece_type == 0:  # pawn
            if white_count > 0 or black_count > 0:
                return False  # 폰이 있으면 기물 부족이 아님
        elif piece_type == 1:  # knight
            white_knights = white_count
            black_knights = black_count
        elif piece_type == 2:  # bishop
            white_bishops = white_count  
            black_bishops = black_count
        elif piece_type == 3 or piece_type == 4:  # rook, queen
            if white_count > 0 or black_count > 0:
                return False  # 룩이나 퀸이 있으면 기물 부족이 아님
                
        white_pieces += white_count
        black_pieces += black_count
    
    # King vs King
    if white_pieces == 1 and black_pieces == 1:
        return True
        
    # King + Bishop vs King 또는 King vs King + Bishop
    if (white_pieces == 2 and black_pieces == 1 and white_bishops == 1) or \
       (white_pieces == 1 and black_pieces == 2 and black_bishops == 1):
        return True
        
    # King + Knight vs King 또는 King vs King + Knight  
    if (white_pieces == 2 and black_pieces == 1 and white_knights == 1) or \
       (white_pieces == 1 and black_pieces == 2 and black_knights == 1):
        return True
        
    return False
