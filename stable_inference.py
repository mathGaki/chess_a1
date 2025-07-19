# src/ai/stable_inference.py
"""
안정적인 추론 서버 (pickle 문제 해결)
"""
import time
import torch
import torch.multiprocessing as mp
from .neural_network import ChessNet
from .config import INFERENCE_TIMEOUT, INFERENCE_BATCH_SIZE
import queue

def stable_inference_server(model_path, device, request_queue, response_queue, stop_event):
    """안정적인 추론 서버"""
    print(f"🚀 Stable inference server starting on {device}")
    
    # 모델 로드
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # TorchScript 최적화
    try:
        model = torch.jit.script(model)
        print("🔥 Model optimized with TorchScript")
    except Exception as e:
        print(f"⚠️ TorchScript failed: {e}")
    
    print(f"🚀 Stable inference server started on {device}")
    
    batch_count = 0
    total_requests = 0
    batch_requests = []
    last_process_time = time.time()
    start_time = time.time()  # 전체 시작 시간 추가
    
    while not stop_event.is_set():
        try:
            # 배치 수집 - config 기반 단순화
            max_collect_time = INFERENCE_TIMEOUT / 10
            collect_start = time.time()
            
            while len(batch_requests) < 512 and (time.time() - collect_start) < max_collect_time:
                try:
                    req = request_queue.get_nowait()
                    batch_requests.append(req)
                except queue.Empty:
                    break
            
            # 실제 텐서 개수 계산
            total_tensors = 0
            for _, _, state in batch_requests:
                if isinstance(state, list):
                    total_tensors += len(state)
                else:
                    total_tensors += 1
            
            # 단순화된 배치 처리 조건
            current_time = time.time()
            time_since_last = current_time - last_process_time
            
            # 배치 처리 조건 (단순화):
            # 1. 텐서 수가 config 배치 크기에 도달하면 즉시 처리
            # 2. 또는 타임아웃에 도달하면 처리 (최소 1개 요청 있을 때)
            should_process = (
                total_tensors >= INFERENCE_BATCH_SIZE or  # config 배치 크기 도달
                (len(batch_requests) > 0 and time_since_last > INFERENCE_TIMEOUT)  # 타임아웃 + 최소 요청
            )
            
            if should_process:
                if batch_requests:
                    # 배치 처리 - 개별 요청과 배치 요청 구분
                    worker_ids, request_ids, states = zip(*batch_requests)
                    
                    # 배치 요청과 개별 요청 분리
                    individual_requests = []
                    batch_requests_list = []
                    
                    for worker_id, request_id, state in zip(worker_ids, request_ids, states):
                        if isinstance(state, list):
                            # 새로운 배치 시스템: state는 텐서 리스트
                            batch_requests_list.append((worker_id, request_id, state))
                        else:
                            # 개별 요청: state는 단일 텐서
                            individual_requests.append((worker_id, request_id, state))
                    
                    # 개별 요청 처리
                    if individual_requests:
                        ind_worker_ids, ind_request_ids, ind_states = zip(*individual_requests)
                        ind_state_batch = torch.stack(ind_states).to(device)
                        
                        with torch.no_grad():
                            ind_policy_logits, ind_values = model(ind_state_batch)
                        
                        # 개별 응답 전송
                        for i, (worker_id, request_id, _) in enumerate(individual_requests):
                            response_queue.put((worker_id, request_id, ind_policy_logits[i].cpu(), ind_values[i].cpu()))
                    
                    # 배치 요청 처리 (새로운 시스템)
                    for worker_id, request_id, batch_tensors in batch_requests_list:
                        if len(batch_tensors) > 0:
                            # 텐서 리스트를 배치로 변환
                            batch_state = torch.stack(batch_tensors).to(device)
                            
                            with torch.no_grad():
                                batch_policy_logits, batch_values = model(batch_state)
                            
                            # 배치 응답을 리스트로 분해해서 전송
                            policies_list = [batch_policy_logits[i].cpu() for i in range(len(batch_tensors))]
                            values_list = [batch_values[i].cpu() for i in range(len(batch_tensors))]
                            
                            response_queue.put((worker_id, request_id, policies_list, values_list))
                    
                    batch_count += 1
                    total_requests += len(batch_requests)
                    
                    # 실제 텐서 개수 계산
                    actual_tensor_count = 0
                    individual_count = 0
                    batch_request_count = 0
                    
                    for _, _, state in batch_requests:
                        if isinstance(state, list):
                            batch_request_count += len(state)
                            actual_tensor_count += len(state)
                        else:
                            individual_count += 1
                            actual_tensor_count += 1
                    
                    # 더 상세한 통계 출력
                    if batch_count % 5 == 0:  # 5배치마다 출력
                        elapsed_time = time.time() - start_time
                        batches_per_sec = batch_count / elapsed_time if elapsed_time > 0 else 0
                        avg_batch_size = total_requests / batch_count if batch_count > 0 else 0
                        avg_tensor_count = actual_tensor_count / len(batch_requests) if len(batch_requests) > 0 else 0
                        
                        print(f"📊 Batch {batch_count}: {total_requests} msgs, {actual_tensor_count} tensors, avg msg/batch: {avg_batch_size:.1f}, avg tensor/msg: {avg_tensor_count:.1f}, {batches_per_sec:.1f} batches/sec")
                    
                    # 배치 크기별 통계
                    if actual_tensor_count >= 500:  # 큰 배치 감지
                        print(f"🚀 Large batch processed: {actual_tensor_count} tensors ({len(batch_requests)} messages) in batch {batch_count}")
                    
                    # 배치 요청 vs 개별 요청 비율
                    if batch_count % 20 == 0:  # 20배치마다 상세 통계
                        print(f"📈 Batch {batch_count}: Individual tensors: {individual_count}, Batch tensors: {batch_request_count}, Total: {actual_tensor_count}")
                    
                    batch_requests = []
                    last_process_time = time.time()
            else:
                time.sleep(0.0001)
                
        except Exception as e:
            print(f"❌ Server error: {e}")
            break
    
    print(f"🔥 Stable inference server stopped. Total: {batch_count} batches, {total_requests} requests")

def stable_selfplay_worker(worker_id, game_queue, request_queue, response_queue, gui_queue=None):
    """안정적인 자기대국 워커"""
    print(f"🚀 Stable Worker {worker_id} starting")
    
    # numba 워밍업
    try:
        from .warmup import warmup_numba_functions
        warmup_numba_functions()
    except Exception as e:
        print(f"⚠️ Worker {worker_id} warmup failed: {e}")
    
    from .game_player import play_game
    from ..core.position import Position, parse_fen
    from ..core.constants import start_position
    from .utils import position_to_tensor
    from .config import MCTS_SIMULATIONS
    
    games_played = 0
    total_time = 0
    
    try:
        while True:
            try:
                game_id = game_queue.get_nowait()
            except queue.Empty:
                break
            
            start_time = time.time()
            
            # 셀프플레이 게임 설정
            game_config = {
                'mode': 'selfplay',
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'temperature': 1.0,
                'dirichlet_noise': True,
                'mcts_simulations': MCTS_SIMULATIONS,
                'request_q': request_queue,
                'response_q': response_queue,
                'max_moves': 200,
                'save_history': True
            }
            
            # 단일 모델로 셀프플레이 (모델은 inference 서버에서 처리)
            models = {'white': None, 'black': None}  # inference 서버 사용
            
            # 실제 게임 플레이
            result = play_game(worker_id, game_id, models, game_config, gui_queue)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            games_played += 1
            
            if games_played % 10 == 0:
                avg_time = total_time / games_played
                print(f"🎮 Worker {worker_id}: {games_played} games, avg {avg_time*1000:.1f}ms/game")
            
            # GUI 업데이트
            if gui_queue and games_played % 5 == 0:
                try:
                    gui_queue.put({'type': 'game_completed', 'worker_id': worker_id, 'game_id': game_id})
                except:
                    pass
                    
    except Exception as e:
        print(f"❌ Worker {worker_id} error: {e}")
    
    print(f"🎯 Stable Worker {worker_id} completed {games_played} games")
