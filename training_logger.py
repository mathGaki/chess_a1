"""
논문용 훈련 통계 로거
학습 과정에서 발생하는 모든 지표를 체계적으로 기록
"""
import os
import json
import csv
import time
from datetime import datetime
from collections import defaultdict
import torch
import numpy as np

class TrainingLogger:
    """논문용 훈련 통계 로거"""
    
    def __init__(self, log_dir='logs', experiment_name=None):
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.exp_dir = os.path.join(log_dir, experiment_name)
        
        # 로그 디렉토리 생성
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 통계 저장소
        self.training_stats = []
        self.iteration_stats = []
        self.inference_stats = []
        self.game_stats = []
        
        # 실시간 누적 통계
        self.current_iteration = 0
        self.total_games = 0
        self.total_training_time = 0
        self.start_time = time.time()
        
        # CSV 파일 헤더 초기화
        self._init_csv_files()
        
        # 실험 설정 저장
        self._save_experiment_config()
    
    def _init_csv_files(self):
        """CSV 파일 헤더 초기화"""
        # 훈련 통계 CSV
        training_csv = os.path.join(self.exp_dir, 'training_stats.csv')
        with open(training_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'step', 'total_loss', 'policy_loss', 'value_loss',
                'learning_rate', 'batch_size', 'training_time', 'timestamp'
            ])
        
        # Iteration 통계 CSV
        iteration_csv = os.path.join(self.exp_dir, 'iteration_stats.csv')
        with open(iteration_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'games_completed', 'avg_game_length', 'win_rate_white',
                'win_rate_black', 'draw_rate', 'selfplay_time', 'training_time',
                'total_time', 'games_per_hour', 'timestamp'
            ])
        
        # 추론 통계 CSV
        inference_csv = os.path.join(self.exp_dir, 'inference_stats.csv')
        with open(inference_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'batch_number', 'batch_size', 'avg_batch_size',
                'batches_per_sec', 'total_requests', 'inference_time_ms',
                'gpu_memory_mb', 'cpu_usage_percent'
            ])
        
        # 게임 통계 CSV
        game_csv = os.path.join(self.exp_dir, 'game_stats.csv')
        with open(game_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'game_id', 'worker_id', 'result', 'move_count',
                'game_time', 'end_reason', 'timestamp'
            ])
    
    def _save_experiment_config(self):
        """실험 설정 저장"""
        from .config import (
            NUM_WORKERS, MCTS_SIMULATIONS, INFERENCE_BATCH_SIZE, INFERENCE_TIMEOUT,
            NN_NUM_CHANNELS, NN_NUM_RES_BLOCKS, learning_rate, weight_decay,
            batch_size, training_steps, ENABLE_TORCH_COMPILE, ENABLE_MIXED_PRECISION
        )
        
        config = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'hardware': {
                'device': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'torch_version': torch.__version__,
            },
            'model_config': {
                'num_channels': NN_NUM_CHANNELS,
                'num_res_blocks': NN_NUM_RES_BLOCKS,
                'enable_torch_compile': ENABLE_TORCH_COMPILE,
                'enable_mixed_precision': ENABLE_MIXED_PRECISION,
            },
            'training_config': {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'training_steps': training_steps,
            },
            'selfplay_config': {
                'num_workers': NUM_WORKERS,
                'mcts_simulations': MCTS_SIMULATIONS,
                'inference_batch_size': INFERENCE_BATCH_SIZE,
                'inference_timeout': INFERENCE_TIMEOUT,
            }
        }
        
        config_file = os.path.join(self.exp_dir, 'experiment_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def log_training_step(self, iteration, step, total_loss, policy_loss, value_loss, 
                         learning_rate, batch_size, training_time):
        """훈련 스텝 통계 기록"""
        stats = {
            'iteration': iteration,
            'step': step,
            'total_loss': float(total_loss),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'training_time': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_stats.append(stats)
        
        # CSV에 실시간 저장
        training_csv = os.path.join(self.exp_dir, 'training_stats.csv')
        with open(training_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['iteration'], stats['step'], stats['total_loss'],
                stats['policy_loss'], stats['value_loss'], stats['learning_rate'],
                stats['batch_size'], stats['training_time'], stats['timestamp']
            ])
    
    def log_iteration_summary(self, iteration, games_completed, game_results, 
                            selfplay_time, training_time):
        """Iteration 요약 통계 기록"""
        # 게임 결과 분석
        total_games = len(game_results)
        if total_games > 0:
            white_wins = sum(1 for r in game_results if r.get('game_result', 0) > 0)
            black_wins = sum(1 for r in game_results if r.get('game_result', 0) < 0)
            draws = sum(1 for r in game_results if r.get('game_result', 0) == 0)
            
            avg_game_length = np.mean([r.get('move_count', 0) for r in game_results])
            win_rate_white = white_wins / total_games
            win_rate_black = black_wins / total_games
            draw_rate = draws / total_games
        else:
            avg_game_length = 0
            win_rate_white = win_rate_black = draw_rate = 0
        
        total_time = selfplay_time + training_time
        games_per_hour = total_games / (total_time / 3600) if total_time > 0 else 0
        
        stats = {
            'iteration': iteration,
            'games_completed': games_completed,
            'avg_game_length': float(avg_game_length),
            'win_rate_white': float(win_rate_white),
            'win_rate_black': float(win_rate_black),
            'draw_rate': float(draw_rate),
            'selfplay_time': float(selfplay_time),
            'training_time': float(training_time),
            'total_time': float(total_time),
            'games_per_hour': float(games_per_hour),
            'timestamp': datetime.now().isoformat()
        }
        
        self.iteration_stats.append(stats)
        
        # CSV에 실시간 저장
        iteration_csv = os.path.join(self.exp_dir, 'iteration_stats.csv')
        with open(iteration_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['iteration'], stats['games_completed'], stats['avg_game_length'],
                stats['win_rate_white'], stats['win_rate_black'], stats['draw_rate'],
                stats['selfplay_time'], stats['training_time'], stats['total_time'],
                stats['games_per_hour'], stats['timestamp']
            ])
    
    def log_inference_batch(self, batch_number, batch_size, avg_batch_size, 
                          batches_per_sec, total_requests, inference_time_ms,
                          gpu_memory_mb=None, cpu_usage_percent=None):
        """추론 배치 통계 기록"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'batch_number': batch_number,
            'batch_size': batch_size,
            'avg_batch_size': float(avg_batch_size),
            'batches_per_sec': float(batches_per_sec),
            'total_requests': total_requests,
            'inference_time_ms': float(inference_time_ms),
            'gpu_memory_mb': gpu_memory_mb,
            'cpu_usage_percent': cpu_usage_percent
        }
        
        self.inference_stats.append(stats)
        
        # CSV에 실시간 저장 (일정 간격마다)
        if batch_number % 10 == 0:  # 10 배치마다 저장
            inference_csv = os.path.join(self.exp_dir, 'inference_stats.csv')
            with open(inference_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    stats['timestamp'], stats['batch_number'], stats['batch_size'],
                    stats['avg_batch_size'], stats['batches_per_sec'], stats['total_requests'],
                    stats['inference_time_ms'], stats['gpu_memory_mb'], stats['cpu_usage_percent']
                ])
    
    def log_game_result(self, iteration, game_id, worker_id, result, move_count, 
                       game_time, end_reason='completed'):
        """개별 게임 결과 기록"""
        stats = {
            'iteration': iteration,
            'game_id': game_id,
            'worker_id': worker_id,
            'result': result,  # 1.0 (white win), -1.0 (black win), 0.0 (draw)
            'move_count': move_count,
            'game_time': float(game_time),
            'end_reason': end_reason,  # 'completed', 'checkmate', 'stalemate', 'draw'
            'timestamp': datetime.now().isoformat()
        }
        
        self.game_stats.append(stats)
        
        # CSV에 실시간 저장 (일정 간격마다)
        if len(self.game_stats) % 50 == 0:  # 50 게임마다 저장
            game_csv = os.path.join(self.exp_dir, 'game_stats.csv')
            with open(game_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for game_stat in self.game_stats[-50:]:  # 마지막 50개 게임만 저장
                    writer.writerow([
                        game_stat['iteration'], game_stat['game_id'], game_stat['worker_id'],
                        game_stat['result'], game_stat['move_count'], game_stat['game_time'],
                        game_stat['end_reason'], game_stat['timestamp']
                    ])
    
    def save_final_summary(self):
        """최종 실험 요약 저장"""
        total_time = time.time() - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_runtime_hours': total_time / 3600,
            'total_iterations': len(self.iteration_stats),
            'total_games': len(self.game_stats),
            'total_training_steps': len(self.training_stats),
            'avg_games_per_hour': len(self.game_stats) / (total_time / 3600) if total_time > 0 else 0,
            'final_stats': {
                'avg_loss': np.mean([s['total_loss'] for s in self.training_stats[-100:]]) if len(self.training_stats) >= 100 else 0,
                'avg_policy_loss': np.mean([s['policy_loss'] for s in self.training_stats[-100:]]) if len(self.training_stats) >= 100 else 0,
                'avg_value_loss': np.mean([s['value_loss'] for s in self.training_stats[-100:]]) if len(self.training_stats) >= 100 else 0,
                'avg_game_length': np.mean([s['move_count'] for s in self.game_stats[-1000:]]) if len(self.game_stats) >= 1000 else 0,
                'final_win_rate': np.mean([1 if s['result'] > 0 else 0 for s in self.game_stats[-1000:]]) if len(self.game_stats) >= 1000 else 0,
            }
        }
        
        summary_file = os.path.join(self.exp_dir, 'final_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Final experiment summary saved to {summary_file}")
        print(f"📈 Total runtime: {total_time/3600:.2f} hours")
        print(f"🎮 Total games: {len(self.game_stats)}")
        print(f"🏋️ Total training steps: {len(self.training_stats)}")
        print(f"📁 All logs saved to: {self.exp_dir}")
    
    def get_current_stats(self):
        """현재 통계 요약 반환"""
        if not self.training_stats:
            return {}
        
        recent_training = self.training_stats[-10:] if len(self.training_stats) >= 10 else self.training_stats
        recent_games = self.game_stats[-100:] if len(self.game_stats) >= 100 else self.game_stats
        
        return {
            'current_iteration': self.current_iteration,
            'total_games': len(self.game_stats),
            'total_training_steps': len(self.training_stats),
            'recent_avg_loss': np.mean([s['total_loss'] for s in recent_training]) if recent_training else 0,
            'recent_avg_game_length': np.mean([s['move_count'] for s in recent_games]) if recent_games else 0,
            'games_per_hour': len(self.game_stats) / ((time.time() - self.start_time) / 3600) if time.time() > self.start_time else 0,
        }

# 전역 로거 인스턴스
_global_logger = None

def get_logger(log_dir='logs', experiment_name=None):
    """전역 로거 인스턴스 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger(log_dir, experiment_name)
    return _global_logger

def set_logger(logger):
    """전역 로거 인스턴스 설정"""
    global _global_logger
    _global_logger = logger
