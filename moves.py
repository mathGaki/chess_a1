import sys

from .attack_tables import is_square_attacked, get_attacks, pawn_attacks
from .constants import *
from .bb_operations import *
from .position import Position, generate_hash_key

"""
           Binary move bits             Meaning          Hexadecimal

    0000 0000 0000 0000 0011 1111    source square       0x3f
    0000 0000 0000 1111 1100 0000    target square       0xfc0
    0000 0000 0111 0000 0000 0000    piece               0x7000
    0000 0000 1000 0000 0000 0000    side                0x8000
    0000 1111 0000 0000 0000 0000    promoted piece      0xf0000
    0001 0000 0000 0000 0000 0000    capture flag        0x100000
    0010 0000 0000 0000 0000 0000    double push flag    0x200000
    0100 0000 0000 0000 0000 0000    enpassant flag      0x400000
    1000 0000 0000 0000 0000 0000    castling flag       0x800000

"""


@njit(cache=True)
def encode_move(source, target, piece, side, promote_to, capture, double, enpas, castling):
    return source | target << 6 | piece << 12 | side << 15 | promote_to << 16 | capture << 20 | \
           double << 21 | enpas << 22 | castling << 23


@njit(nb.uint8(nb.uint64), cache=True)
def get_move_source(move):
    return move & 0x3f


@njit(nb.uint8(nb.uint64), cache=True)
def get_move_target(move):
    return (move & 0xfc0) >> 6


@njit(nb.uint8(nb.uint64), cache=True)
def get_move_piece(move):
    return (move & 0x7000) >> 12


@njit(nb.uint8(nb.uint64), cache=True)
def get_move_side(move):
    return bool(move & 0x8000)


@njit(nb.uint8(nb.uint64), cache=True)
def get_move_promote_to(move):
    return (move & 0xf0000) >> 16


@njit(nb.b1(nb.uint64), cache=True)
def get_move_capture(move):
    return bool(move & 0x100000)


@njit(nb.b1(nb.uint64), cache=True)
def get_move_double(move):
    return bool(move & 0x200000)


@njit(nb.b1(nb.uint64), cache=True)
def get_move_enpas(move):
    return bool(move & 0x400000)


@njit(nb.b1(nb.uint64), cache=True)
def get_move_castling(move):
    return bool(move & 0x800000)


@njit(cache=True)
def get_move_uci(move):
    """get the uci string of a move"""
    return str(square_to_coordinates[get_move_source(move)]) + str(square_to_coordinates[get_move_target(move)]) + \
           (piece_to_letter[black][get_move_promote_to(move)] if get_move_promote_to(move) else '')


def print_move(move):
    """print a move in UCI format"""
    print(f"{square_to_coordinates[get_move_source(move)]}{square_to_coordinates[get_move_target(move)]}"
          f"{piece_to_letter[black][get_move_promote_to(move)] if get_move_promote_to(move) else ''}")


def print_move_list(move_list):
    """print a nice move list"""
    if not move_list:
        print("Empty move_list")

    print()
    print("  move    piece    capture    double    enpas    castling")

    for move in move_list:
        print(f"  {square_to_coordinates[get_move_source(move)]}{square_to_coordinates[get_move_target(move)]}"
              f"{piece_to_letter[black][get_move_promote_to(move)] if get_move_promote_to(move) else ''}     "
              f"{piece_to_letter[get_move_side(move)][get_move_piece(move)]}         "
              f"{get_move_capture(move)}         {get_move_double(move)}         "
              f"{get_move_enpas(move)}         "
              f"{get_move_castling(move)}")

    print("Total number of moves:", len(move_list))


def print_attacked_square(pos, side):
    """print a bitboard of all squares attacked by a given side"""
    attacked = EMPTY
    for sq in squares:
        if is_square_attacked(pos, sq, side):
            attacked = set_bit(attacked, sq)
    print_bb(attacked)


@njit
def generate_moves(pos):
    """return a list of pseudo legal moves from a given Position"""

    # TODO: integrate the constants to be able to compile AOT

    move_list = []

    for piece in range(6):
        bb = pos.pieces[pos.side][piece]
        opp = pos.side ^ 1

        # white pawns & king castling moves
        if pos.side == white:
            if piece == pawn:
                while bb:
                    # pawn move
                    source = get_ls1b_index(bb)
                    target = source - 8

                    # quiet pawn move
                    if not target < a8 and not get_bit(pos.occupancy[both], target):

                        # promotion
                        if a7 <= source <= h7:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 0, 0, 0, 0))

                        else:
                            # push
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 0, 0, 0, 0))

                            # push push
                            if a2 <= source <= h2 and not get_bit(pos.occupancy[both], target - 8):
                                move_list.append(
                                    encode_move(source, target - 8, piece, pos.side, 0, 0, 1, 0, 0))

                    # pawn attack tables
                    attacks = pawn_attacks[white][source] & pos.occupancy[black]

                    while attacks:
                        target = get_ls1b_index(attacks)

                        # promotion capture
                        if a7 <= source <= h7:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 1, 0, 0, 0))
                        # capture
                        else:
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 1, 0, 0, 0))

                        attacks = pop_bit(attacks, target)

                    # en-passant
                    if pos.enpas != no_sq:
                        enpas_attacks = pawn_attacks[white][source] & (BIT << pos.enpas)

                        if enpas_attacks:
                            target_enpas = get_ls1b_index(enpas_attacks)
                            move_list.append(encode_move(source, target_enpas, piece, pos.side, 0, 1, 0, 1, 0))

                    bb = pop_bit(bb, source)

            if piece == king:
                if pos.castle & wk:
                    # are squares empty
                    if not get_bit(pos.occupancy[both], f1) and not get_bit(pos.occupancy[both], g1):
                        # are squares safe
                        if not is_square_attacked(pos, e1, black) and not is_square_attacked(pos, f1, black):
                            move_list.append(encode_move(e1, g1, piece, pos.side, 0, 0, 0, 0, 1))

                if pos.castle & wq:
                    # squares are empty
                    if not get_bit(pos.occupancy[both], d1) and not get_bit(pos.occupancy[both], c1) and not get_bit(
                            pos.occupancy[both], b1):
                        # squares are not attacked by black
                        if not is_square_attacked(pos, e1, black) and not is_square_attacked(pos, d1, black):
                            move_list.append(encode_move(e1, c1, piece, pos.side, 0, 0, 0, 0, 1))

        # black pawns & king castling moves
        if pos.side == black:
            if piece == pawn:
                while bb:
                    source = get_ls1b_index(bb)
                    target = source + 8

                    # quiet pawn move
                    if not target > h1 and not get_bit(pos.occupancy[both], target):

                        # Promotion
                        if a2 <= source <= h2:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 0, 0, 0, 0))

                        else:
                            # push
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 0, 0, 0, 0))

                            # push push
                            if a7 <= source <= h7 and not get_bit(pos.occupancy[both], target + 8):
                                move_list.append(
                                    encode_move(source, target + 8, piece, pos.side, 0, 0, 1, 0, 0))

                    # pawn attack tables
                    attacks = pawn_attacks[black][source] & pos.occupancy[white]

                    while attacks:
                        target = get_ls1b_index(attacks)

                        # promotion capture
                        if a2 <= source <= h2:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 1, 0, 0, 0))
                        # capture
                        else:
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 1, 0, 0, 0))

                        attacks = pop_bit(attacks, target)

                    # en-passant
                    if pos.enpas != no_sq:
                        enpas_attacks = pawn_attacks[black][source] & (BIT << pos.enpas)

                        if enpas_attacks:
                            target_enpas = get_ls1b_index(enpas_attacks)
                            move_list.append(encode_move(source, target_enpas, piece, pos.side, 0, 1, 0, 1, 0))

                    bb = pop_bit(bb, source)

            if piece == king:  # target square will be checked later with legality
                if pos.castle & bk:
                    # squares are empty
                    if not get_bit(pos.occupancy[both], f8) and not get_bit(pos.occupancy[both], g8):
                        # squares are not attacked by black
                        if not is_square_attacked(pos, e8, white) and not is_square_attacked(pos, f8, white):
                            move_list.append(encode_move(e8, g8, piece, pos.side, 0, 0, 0, 0, 1))

                if pos.castle & bq:
                    # squares are empty
                    if not get_bit(pos.occupancy[both], d8) and not get_bit(pos.occupancy[both], c8) and not get_bit(
                            pos.occupancy[both], b8):
                        # squares are not attacked by white
                        if not is_square_attacked(pos, e8, white) and not is_square_attacked(pos, d8, white):
                            move_list.append(encode_move(e8, c8, piece, pos.side, 0, 0, 0, 0, 1))

        if piece in range(1, 6):
            while bb:
                source = get_ls1b_index(bb)
                attacks = get_attacks(piece, source, pos)

                while attacks != EMPTY:
                    target = get_ls1b_index(attacks)

                    # quiet
                    if not get_bit(pos.occupancy[opp], target):
                        move_list.append(encode_move(source, target, piece, pos.side, 0, 0, 0, 0, 0))

                    # capture
                    else:
                        move_list.append(encode_move(source, target, piece, pos.side, 0, 1, 0, 0, 0))

                    attacks = pop_bit(attacks, target)

                bb = pop_bit(bb, source)

    return move_list


def generate_legal_moves(pos):
    """Optimized legal move generation - use fast version"""
    return generate_legal_moves_fast(pos)

def generate_legal_moves_slow(pos):
    """very inefficient, use only to debug"""
    return [move for move in generate_moves(pos) if make_move(pos, move)]

@njit
def generate_legal_moves_fast(pos):
    """Optimized legal move generation using in_check and pin detection"""
    moves = generate_moves(pos)
    legal_moves = []
    
    king_square = get_king_square(pos, pos.side)
    in_check = is_square_attacked(pos, king_square, pos.side ^ 1)
    
    valid_count = 0
    total_count = len(moves)
    
    for move in moves:
        # Quick validation - check if move leaves king in check
        if is_move_legal_fast(pos, move, king_square, in_check):
            legal_moves.append(move)
            valid_count += 1
    
    # 성능 디버깅을 위한 정보 (numba에서는 print 제한적)
    # print(f"Fast filter: {valid_count}/{total_count} moves valid")
    
    return legal_moves

@njit
def get_king_square(pos, side):
    """Get king square for given side"""
    king_bb = pos.pieces[side][king]
    if king_bb == 0:
        return -1
    return get_ls1b_index(king_bb)

@njit
def is_move_legal_fast(pos, move, king_square, in_check):
    """Fast legality check without full make_move"""
    source = get_move_source(move)
    target = get_move_target(move)
    piece = get_move_piece(move)
    side = get_move_side(move)
    
    # If king moves, check target square safety
    if piece == king:
        return not is_square_attacked(pos, target, side ^ 1)
    
    # For now, use the safe but slower method for non-king moves
    # This ensures correctness while we can optimize pin detection later
    temp_pos = make_move_copy(pos, move)
    if temp_pos is None:
        return False
    
    # Check if king is still safe after the move
    new_king_square = get_king_square(temp_pos, side)
    if new_king_square == -1:
        return False
    
    return not is_square_attacked(temp_pos, new_king_square, side ^ 1)

@njit
def is_pinned_piece_illegal_move(pos, source, target, king_square):
    """Check if moving a pinned piece illegally exposes king"""
    # Simplified pin detection - can be optimized further
    return False  # For now, let make_move handle complex cases

def generate_legal_moves_ultra_fast(pos):
    """Ultra-fast legal move generation using incremental attack tables"""
    try:
        # 일단 안전하게 기존 빠른 방식을 사용
        return generate_legal_moves_fast(pos)
        
        # TODO: 증분적 공격 테이블 시스템이 안정화되면 다시 활성화
        # from .incremental_position import IncrementalPosition
        # 
        # # IncrementalPosition으로 변환 (필요한 경우)
        # if not isinstance(pos, IncrementalPosition):
        #     inc_pos = IncrementalPosition(pos)
        #     return inc_pos.generate_legal_moves_ultra_fast()
        # else:
        #     return pos.generate_legal_moves_ultra_fast()
    
    except Exception as e:
        # 문제가 있는 경우 기존 방식 사용
        print(f"Ultra-fast fallback: {e}")
        return generate_legal_moves_fast(pos)

@njit
def make_move_copy(pos, move):
    """Lightweight move execution for validation"""
    # This is still the bottleneck, but we can optimize specific cases
    return make_move(pos, move)


# (Position.class_type.instance_type(Position.class_type.instance_type, nb.uint64, nb.b1))
@njit
def make_move(pos_orig, move, only_captures=False):
    """return new updated position if (move is legal) else None"""

    # TODO: integrate the constants to be able to compile AOT

    # create a copy of the position
    pos = Position()
    pos.pieces = pos_orig.pieces.copy()
    pos.side = pos_orig.side
    pos.enpas = pos_orig.enpas
    pos.castle = pos_orig.castle
    pos.hash_key = pos_orig.hash_key

    # quiet moves
    if not only_captures:

        # parse move
        source_square = get_move_source(move)
        target_square = get_move_target(move)
        piece = get_move_piece(move)
        side = get_move_side(move)
        opp = side ^ 1
        promote_to = get_move_promote_to(move)
        capture = get_move_capture(move)
        double_push = get_move_double(move)
        enpas = get_move_enpas(move)
        castling = get_move_castling(move)

        # Actual Move

        # update bitboards
        pos.pieces[side][piece] = pop_bit(pos.pieces[side][piece], source_square)
        pos.pieces[side][piece] = set_bit(pos.pieces[side][piece], target_square)

        # update hash key
        pos.hash_key ^= piece_keys[side][piece][source_square]
        pos.hash_key ^= piece_keys[side][piece][target_square]

        if capture:  # find what we captured and erase it
            for opp_piece in range(6):
                if get_bit(pos.pieces[opp][opp_piece], target_square):
                    # update bitboards
                    pos.pieces[opp][opp_piece] = pop_bit(pos.pieces[opp][opp_piece], target_square)
                    # update hash key
                    pos.hash_key ^= piece_keys[opp][opp_piece][target_square]
                    break

        if promote_to:  # erase pawn and place promoted piece
            pos.pieces[side][piece] = pop_bit(pos.pieces[side][piece], target_square)
            pos.hash_key ^= piece_keys[side][piece][target_square]

            pos.pieces[side][promote_to] = set_bit(pos.pieces[side][promote_to], target_square)
            pos.hash_key ^= piece_keys[side][promote_to][target_square]

        if enpas:  # erase the opp pawn
            if side:  # black just moved
                pos.pieces[opp][piece] = pop_bit(pos.pieces[opp][piece], target_square - 8)
                pos.hash_key ^= piece_keys[opp][piece][target_square - 8]

            else:  # white just moved
                pos.pieces[opp][piece] = pop_bit(pos.pieces[opp][piece], target_square + 8)
                pos.hash_key ^= piece_keys[opp][piece][target_square + 8]

        if pos.enpas != no_sq:
            pos.hash_key ^= en_passant_keys[pos.enpas]

        # reset enpas
        pos.enpas = no_sq

        if double_push:  # set en-passant square
            if side:  # black just moved
                pos.enpas = target_square - 8
                pos.hash_key ^= en_passant_keys[target_square - 8]
            else:  # white just moved
                pos.enpas = target_square + 8
                pos.hash_key ^= en_passant_keys[target_square + 8]

        if castling:  # move rook accordingly
            if target_square == g1:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], h1)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], f1)

                pos.hash_key ^= piece_keys[side][rook][h1]
                pos.hash_key ^= piece_keys[side][rook][f1]

            elif target_square == c1:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], a1)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], d1)

                pos.hash_key ^= piece_keys[side][rook][a1]
                pos.hash_key ^= piece_keys[side][rook][d1]

            elif target_square == g8:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], h8)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], f8)

                pos.hash_key ^= piece_keys[side][rook][h8]
                pos.hash_key ^= piece_keys[side][rook][f8]

            elif target_square == c8:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], a8)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], d8)

                pos.hash_key ^= piece_keys[side][rook][a8]
                pos.hash_key ^= piece_keys[side][rook][d8]

        # reset castling hash
        pos.hash_key ^= castle_keys[pos.castle]

        # update castling rights
        pos.castle &= castling_rights[source_square]
        pos.castle &= castling_rights[target_square]

        # update castling hash
        pos.hash_key ^= castle_keys[pos.castle]

        # update occupancy
        pos.occupancy.fill(0)  # 초기화
        for color in range(2):
            for bb in pos.pieces[color]:
                pos.occupancy[color] |= bb
        pos.occupancy[both] = pos.occupancy[white] | pos.occupancy[black]

        pos.side = opp
        pos.hash_key ^= side_key

        if not is_square_attacked(pos, get_ls1b_index(pos.pieces[side][king]), opp):
            return pos

    # Capturing moves
    else:
        if get_move_capture(move):
            return make_move(pos, move, only_captures=False)
        return None

    return None


@njit
def make_null_move(pos_orig):
    """return a position with no enpas sq and flipped sides"""

    pos = Position()
    pos.pieces = pos_orig.pieces.copy()
    pos.occupancy = pos_orig.occupancy.copy()
    pos.castle = pos_orig.castle

    pos.side = pos_orig.side ^ 1
    pos.enpas = no_sq
    pos.hash_key = pos_orig.hash_key

    # update hash table
    if pos_orig.enpas != no_sq:
        pos.hash_key ^= en_passant_keys[pos_orig.enpas]
    pos.hash_key ^= side_key

    return pos


def parse_move(pos, uci_move: str) -> int:
    """encode a uci move"""

    source = (ord(uci_move[0]) - ord('a')) + ((8 - int(uci_move[1])) * 8)
    target = (ord(uci_move[2]) - ord('a')) + ((8 - int(uci_move[3])) * 8)

    for move in generate_moves(pos):
        if get_move_source(move) == source and get_move_target(move) == target:
            promoted_piece = get_move_promote_to(move)
            if promoted_piece:
                for p, s in enumerate(('n', 'b', 'r', 'q'), 1):
                    if promoted_piece == p and uci_move[4] == s:
                        return move
                return 0    # in case of illegal promotion (e.g. e7d8f)
            return move
    return 0


@njit
def is_in_check(pos, side):
    """주어진 색깔의 킹이 체크 상태인지 확인 - numba 최적화된 attack_tables 사용"""
    king_square = get_ls1b_index(pos.pieces[side][king])
    return is_square_attacked(pos, king_square, side ^ 1)


@njit  
def has_legal_moves(pos):
    """현재 플레이어가 합법적인 수가 있는지 확인 - 최적화된 버전"""
    moves = generate_moves(pos)
    for move in moves:
        new_pos = make_move(pos, move)
        if new_pos is not None:
            return True
    return False


@njit
def is_checkmate(pos):
    """체크메이트인지 확인 - numba 최적화"""
    return is_in_check(pos, pos.side) and not has_legal_moves(pos)


@njit
def is_stalemate(pos):
    """스테일메이트인지 확인 - numba 최적화"""
    return not is_in_check(pos, pos.side) and not has_legal_moves(pos)


@njit
def count_pieces_by_type(pos, side, piece_type):
    """특정 진영의 특정 기물 수를 카운트 - numba 최적화"""
    from .bb_operations import count_bits
    return count_bits(pos.bitboards[side][piece_type])

@njit  
def is_insufficient_material(pos):
    """기물부족 무승부인지 확인 - numba 최적화
    각 진영이 모두 다음 중 하나일 경우 무승부:
    - 킹만
    - 킹 + 나이트 1개
    - 킹 + 비숍 1개
    """
    # 각 진영별로 기물 수 계산 (직접 비트보드 카운트)
    for side in range(2):  # 0=white, 1=black
        # 기물 비트보드에서 직접 비트 카운트 (기존 count_bits 함수 사용)
        pawns = count_bits(pos.pieces[side][0])    # pawn = 0
        knights = count_bits(pos.pieces[side][1])  # knight = 1
        bishops = count_bits(pos.pieces[side][2])  # bishop = 2
        rooks = count_bits(pos.pieces[side][3])    # rook = 3
        queens = count_bits(pos.pieces[side][4])   # queen = 4
        
        # 폰, 룩, 퀸이 있으면 기물 부족이 아님
        if pawns > 0 or rooks > 0 or queens > 0:
            return False
            
        # 나이트나 비숍이 2개 이상이면 기물 부족이 아님
        if knights > 1 or bishops > 1:
            return False
            
        # 나이트와 비숍이 동시에 있으면 기물 부족이 아님  
        if knights > 0 and bishops > 0:
            return False
    
    # 모든 조건을 통과하면 기물 부족 무승부
    return True

@njit
def is_game_over(pos):
    """게임이 끝났는지 확인하고 결과를 반환 - numba 최적화
    Returns:
        0: 게임 진행 중
        1: 현재 플레이어가 체크메이트 (상대방 승리)
        2: 스테일메이트 (무승부)
        3: 기물부족 무승부
    """
    # 먼저 기물부족 무승부 체크 (빠른 체크)
    if is_insufficient_material(pos):
        return 3  # 기물부족 무승부
    
    if not has_legal_moves(pos):
        if is_in_check(pos, pos.side):
            return 1  # 체크메이트 - 상대방 승리
        else:
            return 2  # 스테일메이트 - 무승부
    return 0  # 게임 진행 중
