# src/selfplay.py
import torch
import torch.multiprocessing as mp
import os
import time
import logging
from datetime import datetime

# 프로젝트의 core 폴더를 경로에 추가해야 함
# 예: sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..core.position import parse_fen
from ..core.moves import generate_moves, make_move, is_game_over, generate_legal_moves, generate_legal_moves_fast, generate_legal_moves_ultra_fast, is_in_check
from ..core.constants import start_position
from ..core.draw_conditions import is_draw, create_position_history, add_to_history, reset_history_on_irreversible_move

from .neural_network import ChessNet
from .stable_inference import stable_inference_server  # InferenceServer → stable_inference_server
from .mcts import MCTS, position_to_tensor
from .config import (GAMES_PER_ITERATION, MCTS_SIMULATIONS, MCTS_C_PUCT, 
                     INFERENCE_BATCH_SIZE, INFERENCE_TIMEOUT, DEVICE, NUM_WORKERS, 
                     INSUFFICIENT_MATERIAL_CHECK_INTERVAL)
from .object_pool import get_position, release_position, get_pool_stats
from .utils import position_to_tensor_cached, get_tensor_cache_stats

def position_to_fen_board(pos):
    """Position 객체를 FEN 보드 부분 문자열로 변환"""
    piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    board = []
    
    for rank in range(8):
        rank_str = ""
        empty_count = 0
        
        for file in range(8):
            square = rank * 8 + file
            piece_found = False
            
            # 각 색깔과 기물 타입 확인
            for color in range(2):
                for piece_type in range(6):
                    if (pos.pieces[color][piece_type] >> square) & 1:
                        if empty_count > 0:
                            rank_str += str(empty_count)
                            empty_count = 0
                        
                        piece_char = piece_chars[color * 6 + piece_type]
                        rank_str += piece_char
                        piece_found = True
                        break
                if piece_found:
                    break
            
            if not piece_found:
                empty_count += 1
        
        if empty_count > 0:
            rank_str += str(empty_count)
        
        board.append(rank_str)
    
    return '/'.join(board)

def selfplay_worker(worker_id, game_queue, request_q, response_q, gui_queue=None):
    """각 워커는 게임 큐에서 작업을 가져와 게임을 진행 (수정된 매개변수 순서)"""
    print(f"Worker {worker_id} started.")
    
    # 워커별 numba 워밍업 (JIT 컴파일)
    try:
        pos = parse_fen(start_position)
        # 각 워커에서 numba 함수 컴파일 (필수!)
        for _ in range(3):
            _ = generate_legal_moves_fast(pos)
        print(f"Worker {worker_id}: numba warmup completed")
    except Exception as e:
        print(f"Worker {worker_id}: numba warmup failed: {e}")
    
    # 각 워커마다 다른 랜덤 시드 설정
    import random
    import numpy as np
    random.seed(worker_id * 12345 + int(time.time()))
    np.random.seed(worker_id * 54321 + int(time.time()))
    torch.manual_seed(worker_id * 98765 + int(time.time()))
    
    # 워커마다 MCTS 인스턴스 생성
    mcts = MCTS(
        inference_queue=request_q,
        result_queue=response_q,
        c_puct=MCTS_C_PUCT
    )
    
    while True:
        try:
            # 게임 큐에서 게임 ID를 가져옴 (timeout을 두어 무한 대기 방지)
            game_id = game_queue.get(timeout=1.0)
        except:
            # 큐가 비어있으면 종료
            print(f"Worker {worker_id} finished - no more games")
            break
        
        print(f"Worker {worker_id} starting game {game_id}")
        
        # 게임 초기화
        pos = parse_fen(start_position)
        history = [] # (state_tensor, policy_target) 저장을 위함
        position_history = create_position_history()  # 포지션 해시 히스토리
        game_result = None  # 게임 결과 초기화

        move_count = 0
        while True: # 게임 루프
            # 현재 포지션을 히스토리에 추가
            add_to_history(position_history, pos)
            
            # GUI 업데이트 (매 수마다) - 워커별로 다른 보드에 표시
            if gui_queue is not None:
                fen_board = position_to_fen_board(pos)
                board_idx = worker_id % 16  # 16개 보드 중 워커별로 할당
                gui_queue.put({'type': 'fen', 'fen': f"{fen_board} {'w' if pos.side == 0 else 'b'} - - 0 1", 'board_id': board_idx})
            
            # 무승부 조건 검사 (스테일메이트, 50수 룰, 3수 동형 반복)
            if is_draw(pos, position_history):
                game_result = 0.0  # 무승부
                print(f"Worker {worker_id} game {game_id}: Draw detected after {move_count} moves")
                break
            
            # MCTS 탐색으로 다음 수와 정책 타겟을 얻음 (원래 방식으로 복원)
            try:
                move, policy_target = mcts.search_simple_batch(pos, generate_legal_moves, MCTS_SIMULATIONS, request_q, response_q, worker_id, move_count, None)
            except Exception as e:
                print(f"Worker {worker_id}: MCTS search failed: {e}")
                break

            # 학습 데이터 저장
            state_tensor = position_to_tensor(pos)
            history.append((state_tensor, policy_target))
            
            # 수 실행
            old_halfmove_clock = pos.halfmove_clock  # 이전 halfmove_clock 저장
            new_pos = make_move(pos, move)
            if new_pos is None: # 만약 make_move가 비합법수라 None을 반환했다면
                print(f"Worker {worker_id}: Invalid move occurred.")
                break
            
            # halfmove_clock이 리셋되었는지 확인 (폰 이동 또는 capture 발생)
            is_irreversible = new_pos.halfmove_clock == 0 and old_halfmove_clock > 0
            
            # 되돌릴 수 없는 수라면 포지션 히스토리 초기화 (50수 룰과 연동)
            if is_irreversible:
                reset_history_on_irreversible_move(position_history, True)
            
            pos = new_pos  # 새 포지션으로 업데이트
            move_count += 1
            
            # 게임 종료 조건 확인 (체크메이트, 스테일메이트, 기물부족 무승부)
            game_status = is_game_over(pos)
            if game_status != 0:
                if game_status == 1:  # 체크메이트 - 이전 플레이어(방금 수를 둔 플레이어) 승리
                    game_result = 1.0 if pos.side == 1 else -1.0  # 현재 턴이 흑이면 백 승리(1.0), 현재 턴이 백이면 흑 승리(-1.0)
                    print(f"Worker {worker_id} game {game_id}: Checkmate! {'White' if game_result > 0 else 'Black'} wins after {move_count} moves")
                elif game_status == 2:  # 스테일메이트
                    game_result = 0.0
                    print(f"Worker {worker_id} game {game_id}: Stalemate after {move_count} moves")
                elif game_status == 3:  # 기물부족 무승부
                    game_result = 0.0
                    print(f"Worker {worker_id} game {game_id}: Insufficient material draw after {move_count} moves")
                break
            
            # 수 제한으로 인한 종료 (체크메이트/스테일메이트가 아닌 경우에만)
            # 워커마다 다른 수로 제한하여 다양성 확보 (150~250수)
            max_moves = 150 + (worker_id * 25) + (game_id % 50)  # 워커와 게임별로 다른 제한
            if move_count > max_moves: 
                print(f"Worker {worker_id} game {game_id} ended after {move_count} moves (limit: {max_moves})")
                game_result = 0.0 # Draw
                break
        
        # 게임 종료 후, 최종 결과를 history의 모든 데이터에 적용하여 저장
        # game_result가 None인 경우 (예외 발생으로 중단) 무승부로 처리
        if game_result is None:
            print(f"Worker {worker_id} game {game_id}: Game ended unexpectedly - treating as draw")
            game_result = 0.0
        
        # 게임 종료 시 최종 GUI 업데이트
        if gui_queue is not None:
            fen_board = position_to_fen_board(pos)
            board_idx = worker_id % 16  # 16개 보드 중 워커별로 할당
            gui_queue.put({'type': 'fen', 'fen': f"{fen_board} {'w' if pos.side == 0 else 'b'} - - 0 1", 'board_id': board_idx})
            
        training_data = []
        for state, policy in history:
            # 현재 플레이어 관점의 결과값
            # (구현의 편의를 위해 단순화. 실제로는 턴에 따라 값을 달리해야 함)
            training_data.append((state, policy, torch.tensor([game_result])))
            
        # 데이터 파일로 저장
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/selfplay/game_w{worker_id}_g{game_id}_{now}.pt"
        torch.save(training_data, filename)
        
        # 게임 결과 통계
        result_str = "Draw"
        if game_result is not None and game_result > 0:
            result_str = "White wins"
        elif game_result is not None and game_result < 0:
            result_str = "Black wins"
        
        print(f"Worker {worker_id}: Game {game_id} completed - {result_str} ({move_count} moves) | Saved to {filename}")
        
        # 주기적으로 최적화 통계 출력 (매 10게임마다)
        if game_id % 10 == 0:
            try:
                pool_stats = get_pool_stats()
                tensor_stats = get_tensor_cache_stats()
                print(f"Worker {worker_id}: Performance Stats - Pool: {pool_stats}, Tensor Cache: {tensor_stats}")
            except:
                pass  # 통계 함수가 없어도 계속 진행

if __name__ == "__main__":
    # 독립 실행용 테스트 코드 (main.py에서는 사용되지 않음)
    mp.set_start_method('spawn') # Numba와 CUDA 사용 시 'spawn'이 안전
    
    if not os.path.exists('data/selfplay'):
        os.makedirs('data/selfplay')
    
    # 간단한 테스트 실행
    print("Selfplay module test mode - use main.py for full training")