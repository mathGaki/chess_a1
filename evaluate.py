import torch
import torch.multiprocessing as mp
import time
import os

from ..core.position import parse_fen
from ..core.moves import generate_moves, make_move, get_move_uci, is_game_over, generate_legal_moves
from ..core.constants import start_position

from .neural_network import ChessNet
from .mcts import MCTS
from .selfplay import position_to_tensor # selfplay에서 유틸리티 함수 임포트
from .config import EVAL_NUM_GAMES, EVAL_MCTS_SIMULATIONS, EVAL_WIN_THRESHOLD, EVAL_NUM_WORKERS, INSUFFICIENT_MATERIAL_CHECK_INTERVAL

# 평가 설정 (AlphaZero 논문 기준)
EVAL_CONFIG = {
    'num_games': EVAL_NUM_GAMES,  # AlphaZero 논문: 400게임
    'mcts_simulations': EVAL_MCTS_SIMULATIONS, # AlphaZero 체스: 800 시뮬레이션
    'c_puct': 1.25,  # 평가시에도 동일한 c_puct 사용
    'win_threshold': EVAL_WIN_THRESHOLD,  # AlphaZero 논문: 55% 승률로 모델 교체
    # 평가 시 완전 결정론적 플레이 (AlphaZero 논문)
    'temperature': 0.0,  # 평가 시 온도 0 (완전 greedy)
    'dirichlet_noise': False,  # 평가 시 Dirichlet noise 비활성화 (논문 기준)
    'add_exploration_noise': False  # 평가 시 탐험 노이즈 없음 (deterministic play)
}

from .game_player import play_game

def evaluate_worker(worker_id, new_model_path, best_model_path, game_queue, result_q, request_q, response_q, gui_queue=None):
    """두 모델을 사용하여 게임 큐에서 동적으로 게임을 가져와 진행하는 워커 (GPU 가속)"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation worker {worker_id} using device: {device}")
    
    # 모델 로드 (GPU 사용)
    new_model = ChessNet()
    new_model.load_state_dict(torch.load(new_model_path, map_location=device))
    new_model.to(device)
    new_model.eval()

    best_model = ChessNet()
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)
    best_model.eval()

    # 모델 딕셔너리 생성
    models = {
        'new': new_model,
        'best': best_model
    }
    
    # 게임 설정 (AlphaZero 논문 기준 - 완전 결정론적 평가)
    game_config = {
        'mode': 'evaluate',
        'device': device,
        'temperature': EVAL_CONFIG['temperature'],  # 0.0 - 평가 시 탐욕적 선택
        'dirichlet_noise': EVAL_CONFIG['dirichlet_noise'],  # False - 평가 시 노이즈 비활성화
        'mcts_simulations': EVAL_CONFIG['mcts_simulations'],  # 평가용 MCTS 시뮬레이션 수
        'request_q': request_q,   # 셀프플레이와 동일한 추론 서버 사용
        'response_q': response_q, # 셀프플레이와 동일한 추론 서버 사용
        'max_moves': 200,
        'save_history': False
    }
    
    print(f"Evaluation worker {worker_id} started.")
    print(f"  - MCTS simulations: {game_config['mcts_simulations']}")
    print(f"  - Temperature: {game_config['temperature']}")
    print(f"  - Dirichlet noise: {game_config['dirichlet_noise']}")
    print(f"  - AlphaZero 논문: Deterministic play for evaluation")

    while True:
        try:
            # 게임 큐에서 게임 ID를 가져옴 (timeout을 두어 무한 대기 방지)
            game_id = game_queue.get(timeout=1.0)
        except:
            # 큐가 비어있으면 종료
            print(f"Evaluation worker {worker_id} finished - no more games")
            break
        
        print(f"Evaluation worker {worker_id} starting game {game_id}")

        # 공통 게임 플레이 함수 사용
        result = play_game(worker_id, game_id, models, game_config, gui_queue)
        
        # 결과 큐에 저장
        result_q.put(result)

def run_evaluation(new_model_path, best_model_path, gui_queue=None):
    """평가 프로세스를 실행하고 결과를 반환 (GPU 가속 및 GUI 연동)"""
    print("--- Starting Evaluation ---")
    print(f"New model: {new_model_path}")
    print(f"Best model: {best_model_path}")
    print(f"Games: {EVAL_CONFIG['num_games']}, Workers: {EVAL_NUM_WORKERS}")

    # 추론 서버를 위한 큐 생성 (셀프플레이와 동일)
    from .stable_inference import stable_inference_server  # InferenceServer → stable_inference_server
    from .config import INFERENCE_BATCH_SIZE, INFERENCE_TIMEOUT
    
    request_q = mp.Queue()
    response_q = mp.Queue()
    
    # 추론 서버 시작 (stable_inference_server 사용)
    model_path = new_model_path  # 이미 저장된 모델 경로 사용
    stop_event = mp.Event()
    
    # stable_inference_server 프로세스 시작
    inference_process = mp.Process(
        target=stable_inference_server,
        args=(model_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
              request_q, response_q, stop_event)
    )
    inference_process.start()
    
    # 게임 큐와 결과 큐 생성 (selfplay와 동일한 방식)
    game_queue = mp.Queue()
    result_q = mp.Queue()
    
    # 게임 큐에 게임 ID들 추가
    for game_id in range(EVAL_CONFIG['num_games']):
        game_queue.put(game_id)

    # 워커 프로세스들 시작 (selfplay와 동일한 동적 할당 방식)
    processes = []
    print(f"🏁 Starting evaluation with {EVAL_NUM_WORKERS} workers...")
    
    for i in range(EVAL_NUM_WORKERS):
        p = mp.Process(target=evaluate_worker, args=(i, new_model_path, best_model_path, game_queue, result_q, request_q, response_q, gui_queue))
        processes.append(p)
        p.start()
        time.sleep(0.1)  # 워커 시작 간격
    
    # 결과 집계 (색깔별 통계 포함) - 워커 종료 전에 먼저 수집
    new_model_wins = 0
    draws = 0
    total_games = 0
    new_white_wins = 0  # 새 모델이 백으로 이긴 횟수
    new_black_wins = 0  # 새 모델이 흑으로 이긴 횟수
    white_draws = 0     # 새 모델이 백일 때 무승부
    black_draws = 0     # 새 모델이 흑일 때 무승부
    
    print("\n--- Collecting Results ---")
    collected_results = []
    
    # 모든 결과를 수집 (워커가 실행 중일 때 수집)
    while total_games < EVAL_CONFIG['num_games']:
        try:
            res = result_q.get(timeout=10.0)  # 10초 timeout (더 여유있게)
            collected_results.append(res)
            total_games += 1
            
            if res['winner'] == 'draw':
                draws += 1
                if res['new_model_color'] == 'white':
                    white_draws += 1
                else:
                    black_draws += 1
                print(f"Game {res['game_id']}: Draw (New model as {res['new_model_color']}, {res['move_count']} moves)")
            elif res['winner'] == res['new_model_color']:
                new_model_wins += 1
                if res['new_model_color'] == 'white':
                    new_white_wins += 1
                else:
                    new_black_wins += 1
                print(f"Game {res['game_id']}: New model wins as {res['new_model_color']} ({res['move_count']} moves)")
            else:
                print(f"Game {res['game_id']}: Best model wins (New model as {res['new_model_color']}, {res['move_count']} moves)")
                
        except Exception as e:
            print(f"Warning: Timeout or error collecting result: {e}")
            print(f"Collected {total_games} results so far, breaking...")
            break
    
    print(f"Collected {total_games} results out of {EVAL_CONFIG['num_games']} expected games")
    
    # 모든 워커 프로세스가 끝날 때까지 대기 (결과 수집 후)
    print("Waiting for all workers to finish...")
    for i, p in enumerate(processes):
        p.join(timeout=30.0)  # 30초 timeout 추가
        if p.is_alive():
            print(f"Warning: Worker {i} is still alive, terminating...")
            p.terminate()
            p.join(timeout=5.0)
    
    print("All workers finished.")
    
    # 추론 서버 정리 (stable_inference_server 방식)
    print("Terminating inference server...")
    stop_event.set()
    inference_process.join(timeout=10.0)
    if inference_process.is_alive():
        print("Warning: Inference server still alive after terminate")
        inference_process.terminate()
    
    # 최종 결과 계산
    best_model_wins = total_games - new_model_wins - draws
    win_rate = (new_model_wins + 0.5 * draws) / total_games if total_games > 0 else 0.0
    
    # 색깔별 승률 계산 (공정성 확인)
    white_games = total_games // 2
    black_games = total_games - white_games
    white_score = new_white_wins + 0.5 * white_draws
    black_score = new_black_wins + 0.5 * black_draws
    white_win_rate = white_score / white_games if white_games > 0 else 0.0
    black_win_rate = black_score / black_games if black_games > 0 else 0.0
    
    print(f"\n--- Evaluation Results (AlphaZero 방식) ---")
    print(f"Total games: {total_games} (백 {white_games}판, 흑 {black_games}판)")
    print(f"New model wins: {new_model_wins} (백으로 {new_white_wins}승, 흑으로 {new_black_wins}승)")
    print(f"Best model wins: {best_model_wins}")  
    print(f"Draws: {draws} (백으로 {white_draws}무, 흑으로 {black_draws}무)")
    print(f"")
    print(f"📊 색깔별 성능:")
    print(f"   백번 승률: {white_win_rate:.2%} ({new_white_wins}승 {white_draws}무 {white_games-new_white_wins-white_draws}패)")
    print(f"   흑번 승률: {black_win_rate:.2%} ({new_black_wins}승 {black_draws}무 {black_games-new_black_wins-black_draws}패)")
    print(f"   전체 승률: {win_rate:.2%} (임계값: {EVAL_CONFIG['win_threshold']:.1%})")
    print(f"")
    print(f"⚙️  평가 설정: MCTS {EVAL_CONFIG['mcts_simulations']}회, 온도={EVAL_CONFIG['temperature']}, 디리클레 노이즈={EVAL_CONFIG['dirichlet_noise']}")
    print(f"   📜 AlphaZero 논문 기준: 완전 결정론적 평가 (deterministic play)")
    print(f"   🎯 승률 임계값: {EVAL_CONFIG['win_threshold']:.1%} (논문 기준: 55%)")

    if win_rate > EVAL_CONFIG['win_threshold']:
        print("✅ New model is better. Updating best model.")
        return True
    else:
        print("❌ New model is not better. Keeping old model.")
        return False

if __name__ == '__main__':
    # 테스트를 위해 가짜 모델 파일 생성
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(ChessNet().state_dict(), 'models/new_model_eval.pth')
    torch.save(ChessNet().state_dict(), 'models/best_model_eval.pth')
    
    run_evaluation('models/new_model_eval.pth', 'models/best_model_eval.pth')
