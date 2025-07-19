"""
효율적인 무승부 조건 검사 모듈
"""

from .position import Position
from .moves import generate_legal_moves, is_in_check
from .constants import *
from numba import njit
import numpy as np

# 해시 히스토리를 저장할 딕셔너리 타입 (numba에서는 typed dict 사용)
from numba.typed import Dict
from numba.core import types

def is_stalemate(pos):
    """
    스테일메이트 검사: 체크가 아니면서 합법수가 없는 상태
    """
    if is_in_check(pos, pos.side):
        return False  # 체크 상태면 스테일메이트가 아님
    
    legal_moves = generate_legal_moves(pos)
    return len(legal_moves) == 0

def is_fifty_move_rule(pos):
    """
    50수 룰: halfmove_clock이 100 이상이면 무승부 (50수 * 2턴)
    """
    return pos.halfmove_clock >= 100

def is_threefold_repetition(pos, position_history):
    """
    3수 동형 반복: 같은 포지션이 3번 나타나면 무승부
    position_history는 hash_key들의 리스트
    """
    current_hash = pos.hash_key
    count = 0
    
    # 현재 포지션 포함하여 카운트
    for hash_key in position_history:
        if hash_key == current_hash:
            count += 1
            if count >= 3:
                return True
    
    return False

def is_draw(pos, position_history=None):
    """
    종합적인 무승부 검사
    
    Args:
        pos: 현재 포지션
        position_history: 이전 포지션들의 해시 리스트 (3수 동형 반복 검사용)
    
    Returns:
        bool: 무승부 조건을 만족하면 True
    """
    # 1. 스테일메이트 검사
    if is_stalemate(pos):
        return True
    
    # 2. 50수 룰 검사
    if is_fifty_move_rule(pos):
        return True
    
    # 3. 3수 동형 반복 검사 (position_history가 제공된 경우만)
    if position_history is not None and is_threefold_repetition(pos, position_history):
        return True
    
    return False

def create_position_history():
    """
    포지션 히스토리를 관리할 리스트 생성
    """
    return []

def add_to_history(position_history, pos):
    """
    포지션 히스토리에 현재 포지션의 해시 추가
    """
    position_history.append(pos.hash_key)
    return position_history

def reset_history_on_irreversible_move(position_history, move_was_irreversible):
    """
    되돌릴 수 없는 수(폰 이동, 기물 capture) 후 히스토리 초기화
    50수 룰과 연동하여 효율성 증대
    """
    if move_was_irreversible:
        position_history.clear()
    return position_history
