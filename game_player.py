"""
공통 게임 플레이 함수 - selfplay와 evaluate에서 공유
"""
import torch
import time
from ..core.position import parse_fen
from ..core.moves import generate_legal_moves, make_move, is_game_over
from ..core.constants import start_position
from .selfplay import position_to_tensor, position_to_fen_board
from .config import INSUFFICIENT_MATERIAL_CHECK_INTERVAL
# from .training_logger import get_logger

class GamePlayer:
    """게임 플레이어 클래스"""
    
    def __init__(self, inference_server, worker_id):
        self.inference_server = inference_server
        self.worker_id = worker_id
        
    def play_game(self, game_id, models, game_config, gui_queue=None):
        """게임 플레이"""
        return play_game(self.worker_id, game_id, models, game_config, gui_queue)

def play_game(worker_id, game_id, models, game_config, gui_queue=None):
    """
    공통 게임 플레이 함수
    
    Args:
        worker_id: 워커 ID
        game_id: 게임 ID
        models: {'white': model, 'black': model} 또는 {'new': model, 'best': model}
        game_config: 게임 설정 딕셔너리
            - mode: 'selfplay' 또는 'evaluate'
            - device: GPU 디바이스
            - mcts_simulations: MCTS 시뮬레이션 수 (selfplay용)
            - temperature: 온도 설정
            - dirichlet_noise: 디리클레 노이즈 사용 여부
            - request_q, response_q: MCTS 큐 (selfplay용)
            - max_moves: 최대 수 제한
            - save_history: 히스토리 저장 여부 (selfplay용)
            
    Returns:
        게임 결과 딕셔너리
    """
    
    device = game_config['device']
    mode = game_config['mode']
    
    # 게임 시작 시간 기록
    game_start_time = time.time()
    
    # 게임 초기화
    pos = parse_fen(start_position)
    move_count = 0
    game_result = None
    winner = None
    history = [] if game_config.get('save_history', False) else None
    
    # 플레이어 할당
    if mode == 'evaluate':
        # 평가 모드: 게임 ID에 따라 색깔 교대
        if game_id % 2 == 0:
            white_model, black_model = models['new'], models['best']
            new_model_color = 'white'
        else:
            white_model, black_model = models['best'], models['new']
            new_model_color = 'black'
    else:
        # 셀프플레이 모드: 같은 모델 (inference 서버 사용)
        white_model = black_model = None  # inference 서버에서 처리
        new_model_color = None
    
    # GUI 초기 상태 업데이트
    if gui_queue:
        board_idx = worker_id % 64  # 64개 보드 사용
        gui_queue.put({'type': 'fen', 'fen': start_position, 'board_id': board_idx})
    
    # 게임 루프
    while True:
        # 현재 플레이어 모델 선택
        current_model = white_model if pos.side == 0 else black_model
        
        # 게임 종료 조건 확인
        game_status = is_game_over(pos)
        if game_status != 0:
            if game_status == 1:  # 체크메이트
                if mode == 'evaluate':
                    winner = 'white' if pos.side == 1 else 'black'
                else:
                    game_result = 1.0 if pos.side == 1 else -1.0
                print(f"Worker {worker_id} game {game_id}: Checkmate! {'White' if (winner == 'white' if mode == 'evaluate' else game_result > 0) else 'Black'} wins after {move_count} moves")
            elif game_status == 2:  # 스테일메이트
                if mode == 'evaluate':
                    winner = 'draw'
                else:
                    game_result = 0.0
                print(f"Worker {worker_id} game {game_id}: Stalemate after {move_count} moves")
            elif game_status == 3:  # 기물부족 무승부
                if mode == 'evaluate':
                    winner = 'draw'
                else:
                    game_result = 0.0
                print(f"Worker {worker_id} game {game_id}: Insufficient material draw after {move_count} moves")
            break
        
        # 합법수 생성
        legal_moves = generate_legal_moves(pos)
        if not legal_moves:
            print(f"Warning: No legal moves but game_status was 0")
            if mode == 'evaluate':
                winner = 'draw'
            else:
                game_result = 0.0
            break
        
        # 수 선택 방식 분기
        if mode == 'selfplay':
            # MCTS 사용 (셀프플레이) - 온도 적응적 적용
            from .mcts import MCTS
            from .config import MCTS_TEMPERATURE_EARLY, MCTS_TEMPERATURE_LATE, MCTS_TEMPERATURE_THRESHOLD
            
            mcts = game_config.get('mcts_instance')
            if mcts is None:
                from .config import MCTS_C_PUCT
                # 큐와 함께 MCTS 인스턴스 생성
                mcts = MCTS(
                    inference_queue=game_config['request_q'],
                    result_queue=game_config['response_q'],
                    c_puct=MCTS_C_PUCT
                )
            
            # 온도 적응적 설정 (AlphaZero 방식)
            if move_count < MCTS_TEMPERATURE_THRESHOLD:
                temp = MCTS_TEMPERATURE_EARLY  # 초기: 탐험적 플레이
            else:
                temp = MCTS_TEMPERATURE_LATE   # 후기: 최적 플레이
            
            try:
                # 새로운 배치 방식 사용 (100개 → 1개 메시지)
                move, policy_target = mcts.search_simple_batch(
                    pos, 
                    generate_legal_moves, 
                    game_config['mcts_simulations'],
                    game_config['request_q'], 
                    game_config['response_q'], 
                    worker_id, 
                    move_count, 
                    None,
                    temperature=temp,
                    add_dirichlet_noise=game_config.get('dirichlet_noise', True)
                )
                
                # 학습 데이터 저장
                if history is not None:
                    state_tensor = position_to_tensor(pos)
                    history.append((state_tensor, policy_target))
                    
            except Exception as e:
                print(f"Worker {worker_id}: MCTS search failed: {e}")
                break
                
        else:  # evaluate 모드 - 셀프플레이와 동일한 MCTS 사용
            # 평가 시에도 MCTS 사용 (셀프플레이와 동일한 추론 서버 사용)
            from .mcts import MCTS
            from .config import EVAL_MCTS_SIMULATIONS
            
            # 평가용 MCTS 인스턴스 생성 (셀프플레이와 동일한 설정)
            mcts = MCTS(
                inference_queue=game_config['request_q'],
                result_queue=game_config['response_q'],
                c_puct=1.25
            )
            
            try:
                # MCTS 탐색 (평가용 설정 - 새로운 배치 방식)
                print(f"Worker {worker_id}: Starting MCTS batch evaluation with {EVAL_MCTS_SIMULATIONS} simulations")
                move, _ = mcts.search_simple_batch(
                    pos, 
                    generate_legal_moves, 
                    EVAL_MCTS_SIMULATIONS,  # 평가용 시뮬레이션 수
                    game_config['request_q'],   # 셀프플레이와 동일한 추론 서버 사용
                    game_config['response_q'],  # 셀프플레이와 동일한 추론 서버 사용
                    worker_id, 
                    move_count, 
                    None,
                    temperature=game_config.get('temperature', 0.0),  # 평가 시 온도 0
                    add_dirichlet_noise=game_config.get('dirichlet_noise', False)  # 평가 시 노이즈 비활성화
                )
                print(f"Worker {worker_id}: MCTS batch evaluation completed, selected move: {move}")
                
            except Exception as e:
                print(f"Worker {worker_id}: MCTS search failed in evaluation: {e}")
                import traceback
                traceback.print_exc()
                
                # MCTS 실패 시 fallback으로 간단한 배치 평가 사용
                print(f"Worker {worker_id}: Falling back to simple batch evaluation")
                if len(legal_moves) == 1:
                    move = legal_moves[0]
                else:
                    next_positions = []
                    valid_moves = []
                    
                    for mv in legal_moves:
                        next_pos = make_move(pos, mv)
                        if next_pos is not None:
                            next_positions.append(next_pos)
                            valid_moves.append(mv)
                    
                    if not valid_moves:
                        winner = 'draw'
                        print(f"Warning: No valid moves after make_move check")
                        break
                    
                    if len(valid_moves) > 1:
                        batch_tensors = [position_to_tensor(next_pos) for next_pos in next_positions]
                        batch_tensor = torch.stack(batch_tensors).to(device)
                        
                        with torch.no_grad():
                            batch_policy, batch_value = current_model(batch_tensor)
                        
                        values = -batch_value.squeeze().cpu().numpy()
                        best_idx = values.argmax()
                        move = valid_moves[best_idx]
                    else:
                        move = valid_moves[0]
        
        # 수 실행
        new_pos = make_move(pos, move)
        if new_pos is None:
            print(f"Worker {worker_id}: Invalid move occurred.")
            break
        
        pos = new_pos
        move_count += 1
        
        # GUI 업데이트
        if gui_queue:
            fen_board = position_to_fen_board(pos)
            board_idx = worker_id % 64  # 64개 보드 사용
            gui_queue.put({'type': 'fen', 'fen': f"{fen_board} {'w' if pos.side == 0 else 'b'} - - 0 1", 'board_id': board_idx})
        
        # 수 제한 확인
        max_moves = game_config.get('max_moves', 200)
        if move_count > max_moves:
            print(f"Worker {worker_id} game {game_id} ended after {move_count} moves (limit: {max_moves})")
            if mode == 'evaluate':
                winner = 'draw'
            else:
                game_result = 0.0
            break
        
        # 주기적 기물부족 무승부 체크
        if move_count % INSUFFICIENT_MATERIAL_CHECK_INTERVAL == 0:
            game_status = is_game_over(pos)
            if game_status == 3:
                if mode == 'evaluate':
                    winner = 'draw'
                else:
                    game_result = 0.0
                print(f"Worker {worker_id} game {game_id}: Insufficient material draw")
                break
    
    # 결과 반환
    game_time = time.time() - game_start_time
    
    if mode == 'evaluate':
        return {
            'game_id': game_id,
            'winner': winner,
            'new_model_color': new_model_color,
            'move_count': move_count,
            'game_time': game_time
        }
    else:
        # 게임 통계 로깅 (비활성화)
        # logger = get_logger()
        current_iteration = game_config.get('iteration', 0)
        
        # 게임 종료 이유 분석
        end_reason = 'completed'
        if game_result is not None:
            if move_count > game_config.get('max_moves', 200):
                end_reason = 'max_moves'
            elif game_result == 0.0:
                end_reason = 'draw'
            else:
                end_reason = 'checkmate'
        
        # logger.log_game_result(
        #     iteration=current_iteration,
        #     game_id=game_id,
        #     worker_id=worker_id,
        #     result=game_result if game_result is not None else 0.0,
        #     move_count=move_count,
        #     game_time=game_time,
        #     end_reason=end_reason
        # )
        
        # 셀프플레이 모드: 훈련 데이터 저장
        if history is not None and game_result is not None:
            from datetime import datetime
            import os
            
            # 훈련 데이터 생성
            training_data = []
            for state, policy_target in history:
                # 현재 턴에 따른 결과값 (백 턴이면 그대로, 흑 턴이면 반대)
                turn_result = game_result if (len(training_data) % 2 == 0) else -game_result
                training_data.append((state, policy_target, torch.tensor([turn_result])))
            
            # 데이터 저장
            os.makedirs('data/selfplay', exist_ok=True)
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/selfplay/game_w{worker_id}_g{game_id}_{now}.pt"
            torch.save(training_data, filename)
            
            # 결과 출력
            result_str = "Draw"
            if game_result > 0:
                result_str = "White wins"
            elif game_result < 0:
                result_str = "Black wins"
            
            print(f"Worker {worker_id}: Game {game_id} completed - {result_str} ({move_count} moves) | Data saved to {filename}")
        
        return {
            'game_id': game_id,
            'game_result': game_result,
            'move_count': move_count,
            'history': history,
            'training_data_saved': history is not None and game_result is not None
        }
