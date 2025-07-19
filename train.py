import os
import glob
import time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import csv
from datetime import datetime
try:
    # PyTorch 2.0+ 새로운 API
    from torch.amp import autocast as torch_autocast, GradScaler as torch_GradScaler
    def get_autocast(device_type='cuda'):
        return torch_autocast(device_type=device_type, dtype=torch.float16)
    def get_gradscaler(device='cuda'):
        return torch_GradScaler(device)
except ImportError:
    # PyTorch 1.x 이전 API
    from torch.cuda.amp import autocast as torch_autocast, GradScaler as torch_GradScaler
    def get_autocast(device_type='cuda'):
        return torch_autocast()
    def get_gradscaler(device='cuda'):
        return torch_GradScaler()

from .neural_network import ChessNet, INPUT_CHANNELS, ACTION_SPACE_SIZE
from .config import learning_rate, weight_decay, batch_size, training_steps, LR_SCHEDULE

class SelfPlayDataset(Dataset):
    """자기 대국 데이터를 로드하기 위한 PyTorch Dataset (최적화된 버전)"""
    def __init__(self, data_dir='data/selfplay'):
        self.data_files = glob.glob(os.path.join(data_dir, '*.pt'))
        self.samples = []
        for file_path in self.data_files:
            # 각 파일은 (state, policy, value) 튜플의 리스트를 포함
            data = torch.load(file_path, map_location='cpu')  # CPU에 먼저 로드
            self.samples.extend(data)
        print(f"Loaded {len(self.data_files)} game files with {len(self.samples)} total positions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, policy_target, value_target = self.samples[idx]
        
        # 메모리 효율성을 위해 float32로 변환 (mixed precision과 호환)
        if state.dtype != torch.float32:
            state = state.float()
        if policy_target.dtype != torch.float32:
            policy_target = policy_target.float()
        if value_target.dtype != torch.float32:
            value_target = value_target.float()
        
        # 데이터 검증 및 클리핑
        state = torch.clamp(state, -10, 10)
        value_target = torch.clamp(value_target, -1, 1)
        
        # Policy target 정규화
        policy_target = policy_target / (policy_target.sum() + 1e-8)
        policy_target = torch.clamp(policy_target, 1e-8, 1.0)
        
        # NaN 체크
        if torch.isnan(state).any() or torch.isnan(policy_target).any() or torch.isnan(value_target).any():
            print(f"⚠️ NaN in data at index {idx}! Using fallback data.")
            # 안전한 대체 데이터 생성
            state = torch.zeros_like(state)
            policy_target = torch.ones_like(policy_target) / len(policy_target)
            value_target = torch.tensor([0.0])
            
        return state, policy_target, value_target

class Trainer:
    """신경망 학습을 담당하는 클래스"""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # 모델 가중치 검증 및 초기화
        print("🔄 Checking model weights...")
        self._verify_model_weights()
        
        # PyTorch 2.0+ 컴파일 최적화 (RTX 5070에서 20-30% 속도 향상)
        if (hasattr(torch, 'compile') and 
            device.type == 'cuda' and 
            config.get('ENABLE_TORCH_COMPILE', True)):
            try:
                # Triton 호환성 체크 및 안전한 컴파일
                test_tensor = torch.randn(1, 1, device=device)
                torch.compile(lambda x: x + 1)(test_tensor)  # 테스트 컴파일
                
                compile_mode = config.get('TORCH_COMPILE_MODE', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                print(f"🚀 Model compiled with PyTorch 2.0+ ({compile_mode}) for faster training")
            except Exception as e:
                print(f"⚠️ Model compilation failed (using standard training): {e}")
                print("💡 Tip: Check if triton-windows is properly installed")
                print("💡 Or set ENABLE_TORCH_COMPILE=False in config.py to disable compilation")
        else:
            print("📝 Using standard PyTorch training (compilation disabled)")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        # AlphaZero 학습률 스케줄러 (RTX 5070에서 더 안정적인 학습)
        from torch.optim.lr_scheduler import PolynomialLR
        self.scheduler = PolynomialLR(self.optimizer, total_iters=config.get('training_steps', 700), power=0.9)
        
        # Mixed Precision Training을 위한 GradScaler (GPU에서만 사용)
        self.use_amp = (device.type == 'cuda' and 
                       config.get('ENABLE_MIXED_PRECISION', True))
        if self.use_amp:
            self.scaler = get_gradscaler('cuda')
            print("🚀 Using Mixed Precision Training (AMP) for faster training")
        else:
            print("⚠️ AMP disabled (CPU mode or disabled in config)")
            self.scaler = None

    def _verify_model_weights(self):
        """모델 가중치가 유효한지 검증하고 문제가 있으면 재초기화"""
        has_nan = False
        has_inf = False
        
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"⚠️ NaN detected in {name}!")
                has_nan = True
            if torch.isinf(param).any():
                print(f"⚠️ Inf detected in {name}!")
                has_inf = True
        
        if has_nan or has_inf:
            print("🔄 Reinitializing model weights...")
            # 모델 재초기화
            for module in self.model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                elif isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            print("✅ Model weights reinitialized")
        else:
            print("✅ Model weights are valid")

    def train(self, dataset):
        """AlphaZero 방식: 에포크 없이 고정된 수의 training steps 실행"""
        print("🔄 Setting model to training mode...")
        self.model.train()
        
        # 논문용 학습 통계 로깅 설정
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/training_log_{timestamp}.csv'
        
        # CSV 파일 헤더 작성
        with open(log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'total_step', 'policy_loss', 'value_loss', 'total_loss', 'learning_rate', 'elapsed_time'])
        
        print(f"📊 Training statistics will be logged to: {log_file}")
        
        # 최적화된 DataLoader 설정 (Windows 호환성 우선)
        num_workers = 0  # 2 → 0 (Windows에서 multiprocessing 데드락 방지)
        print(f"🔄 Creating DataLoader with {num_workers} workers...")
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,  # GPU 전송 속도 향상
            persistent_workers=False,  # Windows 호환성을 위해 비활성화
            prefetch_factor=None if num_workers == 0 else 4,  # num_workers=0일 때 None 필요
            drop_last=True  # 마지막 배치 크기 불일치 방지
        )
        
        print("🔄 Creating infinite dataloader...")
        # 데이터셋을 무한히 반복하기 위한 이터레이터
        from itertools import cycle
        infinite_dataloader = cycle(dataloader)
        
        total_loss = 0
        print(f"Starting training for {self.config['training_steps']} steps...")
        print(f"Current total steps: {self.config.get('total_steps', 0)}")
        print(f"Initial learning rate: {self._get_learning_rate(self.config.get('total_steps', 0)):.6f}")
        print(f"Batch size: {self.config['batch_size']}, Workers: {num_workers}")
        
        # RTX 5070 메모리 관리 최적화
        if self.device.type == 'cuda':
            print("🔄 Setting up GPU optimizations...")
            torch.cuda.empty_cache()  # 시작 전 GPU 메모리 정리
            torch.backends.cudnn.benchmark = True  # cuDNN 자동 최적화
            print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB available")
            
        # 성능 측정 시작
        training_start_time = time.time()
        step_start_time = time.time()
        
        print("🔄 Starting training loop...")
        for step in range(self.config['training_steps']):
            try:
                # AlphaZero 스타일 학습률 스케줄링 (전체 누적 스텝 기준)
                total_step = self.config.get('total_steps', 0) + step
                current_lr = self._get_learning_rate(total_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # 다음 배치 가져오기
                states, policy_targets, value_targets = next(infinite_dataloader)
                states = states.to(self.device, non_blocking=True)  # 비동기 전송
                policy_targets = policy_targets.to(self.device, non_blocking=True)
                value_targets = value_targets.to(self.device, non_blocking=True)

                # 입력 데이터 검증 및 정규화
                if torch.isnan(states).any() or torch.isnan(policy_targets).any() or torch.isnan(value_targets).any():
                    print(f"⚠️ NaN in input data! Skipping batch...")
                    continue
                
                # 입력 데이터 클리핑 (극값 방지)
                states = torch.clamp(states, -10, 10)
                value_targets = torch.clamp(value_targets, -1, 1)
                
                # Policy targets 정규화 (확률 분포 보장)
                policy_targets = policy_targets / (policy_targets.sum(dim=1, keepdim=True) + 1e-8)

                self.optimizer.zero_grad()

                # Mixed Precision Forward Pass
                if self.use_amp:
                    with get_autocast('cuda'):
                        policy_log_preds, value_preds = self.model(states)
                        
                        # Value loss 계산 (클리핑 추가)
                        value_preds = torch.clamp(value_preds, -1, 1)
                        value_loss = F.mse_loss(value_preds.squeeze(), value_targets.squeeze())
                        
                        # Policy loss 계산 (안전한 KL divergence)
                        # log_softmax 결과를 softmax로 변환하여 확률 분포로 만들기
                        policy_probs = F.softmax(policy_log_preds, dim=1)
                        policy_probs = torch.clamp(policy_probs, 1e-8, 1.0)  # 0 방지
                        policy_targets = torch.clamp(policy_targets, 1e-8, 1.0)  # 0 방지
                        
                        # KL divergence 계산 (더 안전한 방식)
                        policy_loss = F.kl_div(torch.log(policy_probs), policy_targets, reduction='batchmean', log_target=False)
                        
                        # NaN 체크 및 처리
                        if torch.isnan(value_loss) or torch.isnan(policy_loss) or torch.isinf(value_loss) or torch.isinf(policy_loss):
                            print(f"⚠️ NaN/Inf detected! value_loss: {value_loss}, policy_loss: {policy_loss}")
                            print(f"   value_preds range: [{value_preds.min():.6f}, {value_preds.max():.6f}]")
                            print(f"   policy_probs range: [{policy_probs.min():.6f}, {policy_probs.max():.6f}]")
                            continue
                        
                        loss = value_loss + policy_loss
                    
                    # Mixed Precision Backward Pass
                    self.scaler.scale(loss).backward()
                    
                    # AlphaZero 원본 방식: L2 정규화만 사용, 그래디언트 클리핑 제거
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # 논문용 통계 로깅 (각 loss 개별 기록)
                    step_log_data = {
                        'step': step + 1,
                        'total_step': self.config.get('total_steps', 0) + step + 1,
                        'policy_loss': policy_loss.item(),
                        'value_loss': value_loss.item(),
                        'total_loss': loss.item(),
                        'learning_rate': current_lr,
                        'elapsed_time': time.time() - step_start_time if 'step_start_time' in locals() else 0
                    }
                    
                    # CSV 파일에 기록
                    with open(log_file, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            step_log_data['step'],
                            step_log_data['total_step'],
                            step_log_data['policy_loss'],
                            step_log_data['value_loss'],
                            step_log_data['total_loss'],
                            step_log_data['learning_rate'],
                            step_log_data['elapsed_time']
                        ])
                else:
                    # 일반 Forward/Backward Pass (CPU 모드)
                    policy_log_preds, value_preds = self.model(states)
                    value_loss = F.mse_loss(value_preds.squeeze(), value_targets.squeeze())
                    
                    # KL divergence 안전 계산 (NaN 방지)
                    policy_loss = F.kl_div(policy_log_preds, policy_targets, reduction='batchmean', log_target=False)
                    
                    # NaN 체크 및 처리
                    if torch.isnan(value_loss) or torch.isnan(policy_loss):
                        print(f"⚠️ NaN detected! value_loss: {value_loss}, policy_loss: {policy_loss}")
                        continue
                    
                    loss = value_loss + policy_loss
                    
                    loss.backward()
                    
                    # AlphaZero 원본 방식: L2 정규화만 사용, 그래디언트 클리핑 제거
                    self.optimizer.step()
                    
                    # 논문용 통계 로깅 (각 loss 개별 기록)
                    step_log_data = {
                        'step': step + 1,
                        'total_step': self.config.get('total_steps', 0) + step + 1,
                        'policy_loss': policy_loss.item(),
                        'value_loss': value_loss.item(),
                        'total_loss': loss.item(),
                        'learning_rate': current_lr,
                        'elapsed_time': time.time() - step_start_time if 'step_start_time' in locals() else 0
                    }
                    
                    # CSV 파일에 기록
                    with open(log_file, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            step_log_data['step'],
                            step_log_data['total_step'],
                            step_log_data['policy_loss'],
                            step_log_data['value_loss'],
                            step_log_data['total_loss'],
                            step_log_data['learning_rate'],
                            step_log_data['elapsed_time']
                        ])
                
                total_loss += loss.item()
                
            except Exception as e:
                print(f"❌ Error at step {step + 1}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 학습률 스케줄러 업데이트
            self.scheduler.step()
            
            # 진행 상황 출력 (매 100 스텝마다)
            if (step + 1) % 100 == 0:
                avg_loss = total_loss / 100
                total_step = self.config.get('total_steps', 0) + step
                elapsed_time = time.time() - step_start_time if 'step_start_time' in locals() else 0
                
                print(f"Step {step + 1}/{self.config['training_steps']} (Total: {total_step + 1}): "
                      f"Loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                      f"Time = {elapsed_time:.2f}s")
                
                total_loss = 0
                step_start_time = time.time()
                
                # RTX 5070 메모리 관리 (매 100 스텝마다)
                if self.device.type == 'cuda' and (step + 1) % 100 == 0:
                    torch.cuda.empty_cache()
        
        # 학습 완료 후 메모리 정리
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # 최종 성능 리포트 (RTX 5070 벤치마킹)
        total_training_time = time.time() - training_start_time
        steps_per_second = self.config['training_steps'] / total_training_time
        time_per_100_steps = (total_training_time / self.config['training_steps']) * 100
        
        final_avg_loss = total_loss / (step + 1) if step > 0 else 0
        print(f"\n✅ Training completed! Final average loss: {final_avg_loss:.6f}")
        print(f"🚀 RTX 5070 Performance Report:")
        print(f"   📊 Total training time: {total_training_time:.2f} seconds")
        print(f"   ⚡ Steps per second: {steps_per_second:.2f}")
        print(f"   ⏱️ Time per 100 steps: {time_per_100_steps:.2f} seconds")
        print(f"   🎯 Target time achieved: {time_per_100_steps < 300:.0f} (< 5 minutes)")
        
        return final_avg_loss
    
    def _get_learning_rate(self, step):
        """AlphaZero 방식의 학습률 스케줄링"""
        for threshold in sorted(LR_SCHEDULE.keys(), reverse=True):
            if step >= threshold:
                return LR_SCHEDULE[threshold]
        return LR_SCHEDULE[0]
            
    def save_model(self, path):
        print(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)

# --- 메인 실행 부분 (테스트용) ---
if __name__ == '__main__':
    # 가상 데이터 생성
    if not os.path.exists('data/selfplay'):
        os.makedirs('data/selfplay')
    
    dummy_data = []
    for _ in range(100):
        state = torch.randn(INPUT_CHANNELS, 8, 8)
        policy = torch.randn(ACTION_SPACE_SIZE)
        policy = F.softmax(policy, dim=0)
        value = torch.tensor([1.0])
        dummy_data.append((state, policy, value))
    torch.save(dummy_data, 'data/selfplay/dummy_game.pt')

    # 학습 설정 (config.py에서 가져오기 - AlphaZero 방식)
    train_config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'training_steps': training_steps  # epochs 대신 training_steps 사용
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet()
    
    dataset = SelfPlayDataset()
    trainer = Trainer(model, device, train_config)
    trainer.train(dataset)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    trainer.save_model('models/trained_model.pth')
