# src/core/incremental_position.py
from .position import Position
from .incremental_attacks import IncrementalAttackTable
from .moves import get_move_source, get_move_target, get_move_piece, get_move_side
from .constants import *

class IncrementalPosition:
    """증분적 공격 테이블을 사용하는 고성능 Position 래퍼 클래스"""
    
    def __init__(self, base_position=None):
        if base_position is None:
            self.pos = Position()
        else:
            # Position 객체를 복사
            self.pos = Position()
            
            # pieces 배열을 안전하게 복사
            import numpy as np
            self.pos.pieces = np.copy(base_position.pieces)
            
            # occupancy를 안전하게 복사 (정수 타입 확인)
            self.pos.occupancy = np.copy(base_position.occupancy)
            
            self.pos.side = int(base_position.side)
            self.pos.enpas = int(base_position.enpas) if base_position.enpas is not None else no_sq
            self.pos.castle = int(base_position.castle)
            self.pos.hash_key = int(base_position.hash_key)
        
        self.attack_table = IncrementalAttackTable()
        self._initialized = False
    
    # Position의 속성들을 위임
    @property
    def pieces(self):
        return self.pos.pieces
    
    @pieces.setter
    def pieces(self, value):
        self.pos.pieces = value
    
    @property
    def side(self):
        return self.pos.side
    
    @side.setter
    def side(self, value):
        self.pos.side = value
    
    @property
    def enpas(self):
        return self.pos.enpas
    
    @enpas.setter
    def enpas(self, value):
        self.pos.enpas = value
    
    @property
    def castle(self):
        return self.pos.castle
    
    @castle.setter
    def castle(self, value):
        self.pos.castle = value
    
    @property
    def hash_key(self):
        return self.pos.hash_key
    
    @hash_key.setter
    def hash_key(self, value):
        self.pos.hash_key = value
    
    def initialize_attacks(self):
        """공격 테이블 초기화 (최초 한 번만)"""
        if not self._initialized:
            self.attack_table.initialize_from_position(self.pos)
            self._initialized = True
    
    def copy(self):
        """포지션 복사 (공격 테이블 포함)"""
        new_pos = IncrementalPosition(self.pos)
        new_pos.attack_table.initialize_from_position(new_pos.pos)
        new_pos._initialized = True
        return new_pos
    
    def make_move_incremental(self, move):
        """증분적 업데이트를 사용한 빠른 move 실행"""
        from .moves import make_move
        
        # 기존 make_move로 새 포지션 생성
        new_pos_basic = make_move(self.pos, move)  # self.pos 사용
        if new_pos_basic is None:
            return None
        
        # IncrementalPosition으로 래핑
        new_pos = IncrementalPosition(new_pos_basic)
        new_pos.attack_table.initialize_from_position(new_pos.pos)
        new_pos._initialized = True
        
        return new_pos
    
    def is_square_attacked_fast(self, square, by_color):
        """빠른 공격 여부 확인"""
        if not self._initialized:
            self.initialize_attacks()
        
        return self.attack_table.is_square_attacked(square, by_color)
    
    def is_king_safe_fast(self, king_color):
        """빠른 킹 안전성 확인"""
        if not self._initialized:
            self.initialize_attacks()
        
        # 킹 위치 찾기
        king_bb = self.pieces[king_color][king]
        if king_bb == 0:
            return False
        
        from .bb_operations import get_ls1b_index
        king_square = get_ls1b_index(king_bb)
        
        return self.attack_table.is_king_safe(king_square, king_color)
    
    def generate_legal_moves_ultra_fast(self):
        """초고속 합법수 생성"""
        from .moves import generate_moves
        
        if not self._initialized:
            self.initialize_attacks()
        
        pseudo_moves = generate_moves(self.pos)  # self.pos 사용
        legal_moves = []
        
        for move in pseudo_moves:
            if self._is_move_legal_ultra_fast(move):
                legal_moves.append(move)
        
        return legal_moves
    
    def _is_move_legal_ultra_fast(self, move):
        """초고속 move 합법성 검사"""
        source = get_move_source(move)
        target = get_move_target(move)
        piece_type = get_move_piece(move)
        side = get_move_side(move)
        
        # 킹 이동의 경우 목표 스퀘어 안전성만 확인
        if piece_type == king:
            return not self.is_square_attacked_fast(target, side ^ 1)
        
        # 다른 기물의 경우 임시 이동 후 킹 안전성 확인
        # (여기서는 단순화 - 실제로는 더 최적화 가능)
        temp_pos = self.make_move_incremental(move)
        if temp_pos is None:
            return False
        
        return temp_pos.is_king_safe_fast(side)


def create_incremental_position_from_fen(fen_string):
    """FEN으로부터 IncrementalPosition 생성"""
    from .position import parse_fen
    
    basic_pos = parse_fen(fen_string)
    inc_pos = IncrementalPosition(basic_pos)
    inc_pos.initialize_attacks()
    
    return inc_pos

def create_incremental_position_from_basic(basic_pos):
    """기본 Position으로부터 IncrementalPosition 생성"""
    inc_pos = IncrementalPosition(basic_pos)
    inc_pos.initialize_attacks()
    
    return inc_pos
