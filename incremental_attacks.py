# src/core/incremental_attacks.py
import numpy as np
from numba import njit
from .constants import *
from .bb_operations import *
from .attack_tables import get_attacks, mask_pawn_attacks, mask_knight_attacks, mask_king_attacks

class IncrementalAttackTable:
    """증분적 공격 테이블 - 변경된 부분만 업데이트"""
    
    def __init__(self):
        # 각 스퀘어별 공격자 집합 (color, piece_type, square)
        self.attackers = [set() for _ in range(64)]
        
        # 각 기물별 현재 공격 범위 캐시
        self.piece_attacks = {}  # (color, piece_type, square) -> set of attacked squares
        
        # 킹 안전성 캐시
        self.king_safety_cache = {}  # (king_square, enemy_color) -> bool
        
        # 더티 플래그 - 업데이트 필요한 스퀘어들
        self.dirty_squares = set()
        self.dirty_king_safety = set()
    
    def initialize_from_position(self, pos):
        """포지션으로부터 초기 공격 테이블 생성"""
        self.clear()
        
        for color in range(2):
            for piece_type in range(6):
                piece_bb = pos.pieces[color][piece_type]
                
                while piece_bb:
                    square = get_ls1b_index(piece_bb)
                    piece_bb = pop_bit(piece_bb, square)
                    
                    # 해당 기물의 공격 범위 계산 및 추가
                    self._add_piece_attacks(pos, color, piece_type, square)
    
    def clear(self):
        """모든 캐시 초기화"""
        for i in range(64):
            self.attackers[i].clear()
        self.piece_attacks.clear()
        self.king_safety_cache.clear()
        self.dirty_squares.clear()
        self.dirty_king_safety.clear()
    
    def _add_piece_attacks(self, pos, color, piece_type, square):
        """특정 기물의 공격 범위를 테이블에 추가"""
        attacked_squares = self._calculate_piece_attacks(pos, color, piece_type, square)
        
        # 공격 범위 캐시에 저장
        piece_key = (color, piece_type, square)
        self.piece_attacks[piece_key] = attacked_squares
        
        # 각 공격 대상 스퀘어에 공격자 정보 추가
        for target_sq in attacked_squares:
            self.attackers[target_sq].add(piece_key)
    
    def _remove_piece_attacks(self, color, piece_type, square):
        """특정 기물의 공격 범위를 테이블에서 제거"""
        piece_key = (color, piece_type, square)
        
        if piece_key in self.piece_attacks:
            attacked_squares = self.piece_attacks[piece_key]
            
            # 각 공격 대상 스퀘어에서 공격자 정보 제거
            for target_sq in attacked_squares:
                self.attackers[target_sq].discard(piece_key)
            
            # 캐시에서 제거
            del self.piece_attacks[piece_key]
    
    def _calculate_piece_attacks(self, pos, color, piece_type, square):
        """특정 기물의 공격 범위 계산"""
        attacked_squares = set()
        
        if piece_type == pawn:
            # 폰 공격 (대각선만)
            attacks_bb = mask_pawn_attacks(color, square)
            attacked_squares = self._bb_to_square_set(attacks_bb)
            
        elif piece_type == knight:
            # 나이트 공격
            attacks_bb = mask_knight_attacks(square)
            attacked_squares = self._bb_to_square_set(attacks_bb)
            
        elif piece_type == king:
            # 킹 공격
            attacks_bb = mask_king_attacks(square)
            attacked_squares = self._bb_to_square_set(attacks_bb)
            
        elif piece_type in [bishop, rook, queen]:
            # 슬라이딩 기물 - Position 객체 전체를 전달
            attacks_bb = get_attacks(piece_type, square, pos)
            attacked_squares = self._bb_to_square_set(attacks_bb)
        
        return attacked_squares
    
    def _bb_to_square_set(self, bb):
        """비트보드를 스퀘어 집합으로 변환"""
        squares = set()
        while bb:
            square = get_ls1b_index(bb)
            squares.add(square)
            bb = pop_bit(bb, square)
        return squares
    
    def _get_occupancy(self, pos):
        """현재 보드의 점유 비트보드 반환"""
        occupancy = 0
        for color in range(2):
            for piece_type in range(6):
                occupancy |= pos.pieces[color][piece_type]
        return occupancy
    
    def update_move(self, pos, from_sq, to_sq, piece_type, color, captured_piece=None):
        """기물 이동에 따른 증분적 업데이트"""
        # 1. 이동한 기물의 이전 공격 범위 제거
        self._remove_piece_attacks(color, piece_type, from_sq)
        
        # 2. 캡처된 기물이 있다면 그 공격 범위도 제거
        if captured_piece is not None:
            captured_color = color ^ 1
            self._remove_piece_attacks(captured_color, captured_piece, to_sq)
        
        # 3. 이동한 기물의 새로운 공격 범위 추가
        self._add_piece_attacks(pos, color, piece_type, to_sq)
        
        # 4. 슬라이딩 기물들의 공격 범위 업데이트 (라인이 변경된 경우)
        self._update_sliding_pieces_on_line(pos, from_sq, to_sq)
        
        # 5. 킹 안전성 캐시 무효화
        self._invalidate_king_safety_cache()
    
    def _update_sliding_pieces_on_line(self, pos, from_sq, to_sq):
        """from_sq와 to_sq 라인상의 슬라이딩 기물들 업데이트"""
        # 수직/수평/대각선 라인상의 기물들을 찾아서 업데이트
        affected_squares = self._get_line_squares(from_sq, to_sq)
        
        for sq in affected_squares:
            for color in range(2):
                # 룩, 비숍, 퀸 확인
                for piece_type in [rook, bishop, queen]:
                    if (pos.pieces[color][piece_type] >> sq) & 1:
                        # 해당 기물의 공격 범위 재계산
                        self._remove_piece_attacks(color, piece_type, sq)
                        self._add_piece_attacks(pos, color, piece_type, sq)
    
    def _get_line_squares(self, from_sq, to_sq):
        """두 스퀘어 사이의 라인상에 있는 모든 스퀘어 반환"""
        # 간단한 구현 - 실제로는 더 최적화 가능
        affected = set()
        
        # 수직 라인
        file_from, file_to = from_sq % 8, to_sq % 8
        rank_from, rank_to = from_sq // 8, to_sq // 8
        
        if file_from == file_to:  # 같은 파일
            for rank in range(8):
                affected.add(rank * 8 + file_from)
        
        if rank_from == rank_to:  # 같은 랭크
            for file in range(8):
                affected.add(rank_from * 8 + file)
        
        # 대각선도 추가 가능하지만 복잡함
        
        return affected
    
    def _invalidate_king_safety_cache(self):
        """킹 안전성 캐시 무효화"""
        self.king_safety_cache.clear()
        self.dirty_king_safety.clear()
    
    def is_square_attacked(self, square, by_color):
        """특정 스퀘어가 특정 색에 의해 공격받는지 확인"""
        return any(
            attacker[0] == by_color 
            for attacker in self.attackers[square]
        )
    
    def is_king_safe(self, king_square, king_color):
        """킹이 안전한지 확인 (캐시 사용)"""
        enemy_color = king_color ^ 1
        cache_key = (king_square, enemy_color)
        
        if cache_key not in self.king_safety_cache:
            self.king_safety_cache[cache_key] = not self.is_square_attacked(king_square, enemy_color)
        
        return self.king_safety_cache[cache_key]
    
    def get_attackers(self, square, by_color=None):
        """특정 스퀘어의 공격자 목록 반환"""
        if by_color is None:
            return list(self.attackers[square])
        else:
            return [
                attacker for attacker in self.attackers[square]
                if attacker[0] == by_color
            ]


# 전역 증분적 공격 테이블 인스턴스
_global_attack_table = IncrementalAttackTable()

def get_global_attack_table():
    """전역 공격 테이블 반환"""
    return _global_attack_table

def reset_global_attack_table():
    """전역 공격 테이블 초기화"""
    _global_attack_table.clear()
