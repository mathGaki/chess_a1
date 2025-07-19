# src/mcts.py
import numpy as np
import torch
import math
import copy
from .utils import move_to_policy_index, policy_index_to_move, position_to_tensor, position_to_tensor_cached
from .config import (ACTION_SPACE_SIZE, MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_WEIGHT, 
                     MCTS_TEMPERATURE_EARLY, MCTS_TEMPERATURE_LATE, MCTS_TEMPERATURE_THRESHOLD,
                     MCTS_C_PUCT)
from ..core.constants import start_position
from ..core.moves import make_move

class Node:
    """MCTS 트리의 노드 (메모리 최적화 버전)"""
    __slots__ = ['parent', 'children', 'visit_count', 'total_action_value', 
                 'mean_action_value', 'prior_p', 'position']
    
    def __init__(self, parent, prior_p, position=None):
        self.parent = parent
        self.children = {}  # {move: Node}
        self.visit_count = 0
        self.total_action_value = 0.0
        self.mean_action_value = 0.0
        self.prior_p = prior_p
        self.position = position # 이 노드가 나타내는 게임 상태

    def select(self, c_puct, logger=None, depth=0):
        """UCT(PUCT) 점수가 가장 높은 자식 노드 선택 (최적화된 버전)"""
        if not self.children:
            return None, None
            
        best_score = -float('inf')
        best_move = None
        best_child = None

        # 제곱근 미리 계산 (반복 계산 방지)
        sqrt_parent_visits = math.sqrt(self.visit_count)
        c_puct_sqrt = c_puct * sqrt_parent_visits  # 공통 계산 미리 수행
        
        # 최적화된 반복문 (안전성 체크 포함)
        for move, child in self.children.items():
            # 안전성 체크: position이 None인 자식 노드는 건너뜀
            if child.position is None:
                continue
                
            # UCT 점수 계산 (AlphaZero PUCT 공식) - 수학적으로 동일
            if child.visit_count == 0:
                # 미방문 노드: P(s,a) * c_puct * sqrt(N(s))
                uct_score = child.prior_p * c_puct_sqrt
            else:
                # 방문된 노드: Q(s,a) + P(s,a) * c_puct * sqrt(N(s)) / (1 + N(s,a))
                uct_score = child.mean_action_value + (child.prior_p * c_puct_sqrt) / (1 + child.visit_count)
            
            if uct_score > best_score:
                best_score = uct_score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, legal_moves, policy):
        """
        리프 노드를 확장. 자식 노드들을 생성합니다.
        AlphaZero 논문 (Methods -> Search)에 따라, 신경망의 정책(p)을 사용하여
        각 유효한 수에 대한 사전 확률을 설정합니다.
        
        Args:
            legal_moves: 합법수 리스트
            policy: 합법수에 대응하는 정책 확률 배열 (len(legal_moves)와 같은 크기)
        """
        
        for i, move in enumerate(legal_moves):
            if move not in self.children:
                # policy는 합법수 순서대로 정렬된 확률 배열
                try:
                    prior_prob = policy[i].item() if isinstance(policy, torch.Tensor) else policy[i]
                except IndexError:
                    print(f"Policy index {i} out of bounds (policy size: {len(policy)}, legal_moves: {len(legal_moves)})")
                    prior_prob = 1.0 / len(legal_moves)  # 균등 분포로 기본값
                
                # 자식 노드의 position은 현재 노드의 position에 해당 move를 적용한 결과입니다.
                next_position = make_move(self.position, move)
                
                # make_move가 None을 반환하는 경우 (불법수) 자식 노드를 생성하지 않음
                if next_position is not None:
                    self.children[move] = Node(parent=self, prior_p=prior_prob, position=next_position)
                # make_move가 None을 반환하는 경우는 조용히 무시 (불법수)
    
    def backpropagate(self, value):
        """
        리프 노드의 평가값(value)을 루트까지 역전파합니다. (최적화된 버전)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent
            value = -value  # AlphaZero: 상대방 시점으로 변환
    
    def is_fully_expanded(self):
        """
        노드가 완전히 확장되었는지 확인합니다.
        """
        return len(self.children) > 0 if self.position is not None else True
    
    def is_leaf(self):
        """리프 노드인지 확인합니다."""
        return len(self.children) == 0

class MCTS:
    """몬테카를로 트리 탐색 클래스"""
    def __init__(self, c_puct=None):
        self.c_puct = c_puct if c_puct is not None else MCTS_C_PUCT
        self.root = None

    def search(self, position, move_generator, num_simulations, req_q, resp_q, worker_id, move_count=0, logger=None, temperature=1.0, add_dirichlet_noise=True, dirichlet_alpha=None):
        """
        MCTS 탐색을 수행하여 다음 수를 결정하고 정책 타겟을 생성합니다.
        AlphaZero 논문 (Methods -> Search)의 MCTS 과정을 따릅니다.
        
        Args:
            position (Position): 현재 게임 상태.
            move_generator (function): 현재 상태에서 가능한 유효한 수들을 반환하는 함수.
                                       (예: legal_moves = move_generator(current_position))
            num_simulations (int): 수행할 시뮬레이션 횟수.
            req_q (Queue): 신경망 평가 요청을 보내는 큐. (worker_id, request_id, tensor)
            resp_q (Queue): 신경망 평가 응답을 받는 큐. (worker_id, request_id, policy, value)
            worker_id (int): 현재 MCTS 워커의 ID.
            move_count (int): 현재 수 번호.
            logger: 로거 (선택사항).
            temperature (float): 수 선택 시 온도 매개변수 (기본값: 1.0).
            add_dirichlet_noise (bool): 루트에서 Dirichlet noise 추가 여부 (기본값: True).
        
        Returns:
            tuple: (best_move, policy_target)
                   best_move: 탐색 결과 가장 좋은 수.
                   policy_target: 각 수의 방문 횟수 분포 (신경망 학습을 위한 타겟).
        """
        # 루트 노드 생성 및 초기 평가 요청 (캐시된 텐서 사용)
        # 루트 노드는 현재 게임 상태를 가지고 시작합니다.
        root_tensor = position_to_tensor_cached(position)
        
        req_q.put((worker_id, 0, root_tensor))
        
        # 루트 평가 응답 대기
        policy, value = self._wait_for_response(0, resp_q, worker_id)
        
        legal_moves = move_generator(position)
        
        # 루트 노드의 prior_p는 사용되지 않으므로 0으로 설정하거나 다른 의미 없는 값으로 설정합니다.
        self.root = Node(parent=None, prior_p=0, position=position)
        
        # 합법수에 해당하는 정책 확률만 추출
        legal_move_policies = []
        for move in legal_moves:
            try:
                move_idx = move_to_policy_index(move, position.side)
                prob = policy[move_idx].item() if isinstance(policy, torch.Tensor) else policy[move_idx]
                legal_move_policies.append(prob)
            except Exception as e:
                print(f"Error extracting policy for move {move}: {e}")
                legal_move_policies.append(1.0 / len(legal_moves))  # 균등 분포
        
        # 정규화 (확률의 합이 1이 되도록)
        if len(legal_move_policies) > 0:
            policy_sum = sum(legal_move_policies)
            if policy_sum > 0:
                legal_move_policies = [p / policy_sum for p in legal_move_policies]
            else:
                legal_move_policies = [1.0 / len(legal_move_policies) for _ in legal_move_policies]
        
        # AlphaZero: 루트에서만 Dirichlet noise 추가 (조건부)
        if add_dirichlet_noise:
            alpha = dirichlet_alpha if dirichlet_alpha is not None else MCTS_DIRICHLET_ALPHA
            noise = np.random.dirichlet([alpha] * len(legal_move_policies))
            noise_weight = MCTS_DIRICHLET_WEIGHT
            legal_move_policies_with_noise = (1 - noise_weight) * np.array(legal_move_policies) + noise_weight * noise
        else:
            legal_move_policies_with_noise = legal_move_policies
        
        self.root.expand(legal_moves, legal_move_policies_with_noise)
        self.root.backpropagate(value.item())

        # 시뮬레이션 반복 (순수 AlphaZero 방식)
        for i in range(1, num_simulations + 1):
            node = self.root
            path = [node] # 탐색 경로를 저장하여 역전파에 사용

            # 1. Select: PUCT를 사용하여 리프 노드까지 탐색 (순수 AlphaZero)
            depth = 0
            while not node.is_leaf():
                move, node = node.select(self.c_puct, None, depth)
                
                # select가 None을 반환하면 시뮬레이션 중단
                if move is None or node is None:
                    print(f"Warning: select() returned None in simulation {i}, depth {depth}")
                    break
                    
                path.append(node)
                depth += 1
            
            # 리프 노드의 게임 상태 (sim_pos)
            # path의 마지막 노드가 현재 시뮬레이션의 리프 노드이므로, 해당 노드의 position을 사용합니다.
            sim_pos = node.position 
            
            # None position 체크
            if sim_pos is None:
                print(f"Warning: sim_pos is None in MCTS simulation {i}")
                continue
            
            # 2. Expand & Evaluate: 리프 노드 확장 및 신경망 평가 요청 (캐시된 텐서 사용)
            # 리프 노드의 상태를 텐서로 변환하여 신경망에 평가 요청을 보냅니다.
            leaf_pos_tensor = position_to_tensor_cached(sim_pos) 
            req_q.put((worker_id, i, leaf_pos_tensor))
            
            # 3. Backpropagate (응답 대기 후): 신경망 평가 결과로 역전파
            policy, value = self._wait_for_response(i, resp_q, worker_id)
            
            # 리프 노드에서 가능한 유효한 수들을 생성합니다.
            legal_moves_at_leaf = move_generator(sim_pos)
            
            # 합법수에 해당하는 정책 확률만 추출
            legal_move_policies = []
            for move in legal_moves_at_leaf:
                try:
                    move_idx = move_to_policy_index(move, sim_pos.side)
                    prob = policy[move_idx].item() if isinstance(policy, torch.Tensor) else policy[move_idx]
                    legal_move_policies.append(prob)
                except Exception as e:
                    print(f"Error extracting policy for move {move}: {e}")
                    legal_move_policies.append(1.0 / len(legal_moves_at_leaf))  # 균등 분포
            
            # 정규화 (확률의 합이 1이 되도록)
            if len(legal_move_policies) > 0:
                policy_sum = sum(legal_move_policies)
                if policy_sum > 0:
                    legal_move_policies = [p / policy_sum for p in legal_move_policies]
                else:
                    legal_move_policies = [1.0 / len(legal_move_policies) for _ in legal_move_policies]
            
            node.expand(legal_moves_at_leaf, legal_move_policies) # 리프 노드 확장 (합법수 정책만 전달)
            node.backpropagate(value.item()) # 평가값 역전파

        # 탐색 완료 후 루트의 방문 횟수를 기반으로 다음 수 선택
        move_counts = []
        moves = []
        move_scores = []  # UCT 점수 기록을 위해 추가
        
        for move, child in self.root.children.items():
            moves.append(move)
            move_counts.append(child.visit_count)
            # 일관된 UCT 점수 계산 (select 메서드와 동일한 로직)
            sqrt_parent_visits = math.sqrt(self.root.visit_count)
            
            if child.visit_count == 0:
                # 미방문 노드: prior_p를 기준으로 계산
                uct_score = self.c_puct * child.prior_p * sqrt_parent_visits
            else:
                q_value = child.mean_action_value
                u_value = self.c_puct * child.prior_p * sqrt_parent_visits / (1 + child.visit_count)
                uct_score = q_value + u_value
                
            move_scores.append(uct_score)
        
        if not moves:
            print(f"Warning: No moves available from root!")
            return None, torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
        
        # AlphaZero 방식: 온도 기반 선택 (매개변수 사용)
        if temperature == 0.0:
            # 탐욕적 선택 (가장 많이 방문된 수)
            best_move = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
            selected_idx = moves.index(best_move)
            selection_reason = "greedy (temp=0)"
        else:
            # 온도 기반 확률적 선택
            move_probs = np.array(move_counts, dtype=np.float64)
            
            # AlphaZero 논문의 온도 공식: π(a) ∝ N(s,a)^(1/τ)
            move_probs = move_probs ** (1.0 / temperature)
            move_probs = move_probs / np.sum(move_probs)
            
            # 확률적 선택 (AlphaZero 방식)
            selected_idx = np.random.choice(len(moves), p=move_probs)
            best_move = moves[selected_idx]
            selection_reason = f"temperature-based (temp={temperature:.2f})"
        
        # 디버그 로그 (평가 모드에서만)
        if temperature == 0.0:  # 평가 모드
            # 상위 3개 수의 방문 횟수 표시
            move_visit_pairs = list(zip(moves, move_counts))
            move_visit_pairs.sort(key=lambda x: x[1], reverse=True)
            top_moves = move_visit_pairs[:3]
            top_moves_str = ", ".join([f"{move}({visits})" for move, visits in top_moves])
            print(f"  MCTS worker {worker_id}: {num_simulations} sims, selected {selection_reason}")
            print(f"    Top moves: {top_moves_str}, selected: {best_move}({move_counts[selected_idx]})")
            
            # 전체 방문 분포 (방문 횟수 > 0인 수들만)
            visited_moves = [(move, count) for move, count in zip(moves, move_counts) if count > 0]
            if len(visited_moves) > 1:
                print(f"    Total visited moves: {len(visited_moves)}/{len(moves)}")
            else:
                print(f"    Warning: Only {len(visited_moves)} moves visited out of {len(moves)} legal moves!")
        
        # 정책 타겟 생성 (방문 횟수 분포)
        # 신경망 학습을 위한 타겟으로, 루트 노드의 자식 방문 횟수를 기반으로 정책 분포를 만듭니다.
        policy_target = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32) 
        total_visits = sum(child.visit_count for child in self.root.children.values())
        
        if total_visits > 0:
            for move, child in self.root.children.items():
                move_idx = move_to_policy_index(move, self.root.position.side) 
                policy_target[move_idx] = child.visit_count / total_visits

        return best_move, policy_target

    def search_simple_batch(self, position, move_generator, num_simulations, req_q, resp_q, worker_id, move_count, logger=None, 
                           temperature=None, add_dirichlet_noise=True, dirichlet_alpha=None):
        """
        하나의 배치 메시지로 모든 시뮬레이션을 처리하는 방식 (디버깅 모드)
        """
        # print(f"🔍 Worker {worker_id}: Starting MCTS with {num_simulations} simulations")
        
        # 1. 루트 노드 초기화
        root_tensor = position_to_tensor_cached(position)
        # print(f"🔍 Worker {worker_id}: Sending root request")
        req_q.put((worker_id, "root", root_tensor))
        
        # print(f"🔍 Worker {worker_id}: Waiting for root response")
        policy, value = self._wait_for_batch_response("root", resp_q, worker_id)
        # print(f"🔍 Worker {worker_id}: Got root response, policy shape: {policy.shape}, value: {value.item()}")
        
        legal_moves = move_generator(position)
        # print(f"🔍 Worker {worker_id}: Found {len(legal_moves)} legal moves")
        
        self.root = Node(parent=None, prior_p=0, position=position)
        
        # 루트 확장
        legal_move_policies = self._extract_legal_policies(legal_moves, policy, position.side)
        if add_dirichlet_noise:
            legal_move_policies = self._add_dirichlet_noise(legal_move_policies, dirichlet_alpha)
        
        self.root.expand(legal_moves, legal_move_policies)
        self.root.backpropagate(value.item())
        # print(f"🔍 Worker {worker_id}: Root expanded with {len(self.root.children)} children")
        
        # 2. 모든 시뮬레이션의 리프 노드를 미리 수집 (최적화된 버전)
        all_leaf_nodes = []
        all_leaf_paths = []
        
        # 최적화: 반복 탐색을 위한 캐시된 변수들
        max_depth = 200  # 무한 루프 방지
        
        for sim_idx in range(num_simulations):
            # 각 시뮬레이션의 리프 노드까지 탐색 (최적화된 버전)
            path = []
            current = self.root
            depth = 0
            
            # 최적화된 탐색 루프
            while current.children and depth < max_depth:
                move, next_node = current.select(self.c_puct)
                if next_node is None:
                    break
                current = next_node
                path.append(current)
                depth += 1
            
            # 유효한 노드만 추가 (None 체크 최소화)
            if current.position is not None:
                all_leaf_nodes.append(current)
                all_leaf_paths.append(path)
        
        # 배치 효율성 체크 (최적화된 버전)
        collected_count = len(all_leaf_nodes)
        if collected_count < num_simulations * 0.8:  # 80% 미만이면 경고
            efficiency = collected_count / num_simulations * 100
            print(f"⚠️ Worker {worker_id}: Low batch efficiency - {collected_count}/{num_simulations} simulations ({efficiency:.1f}%)")
        
        # 3. 모든 리프 노드를 하나의 배치로 묶어서 전송
        if all_leaf_nodes:
            batch_size = len(all_leaf_nodes)
            # print(f"🔍 Worker {worker_id}: Sending batch message with {batch_size} tensors")
            
            # 모든 텐서를 리스트로 묶어서 하나의 메시지로 전송
            batch_tensors = []
            for i, node in enumerate(all_leaf_nodes):
                leaf_tensor = position_to_tensor_cached(node.position)
                batch_tensors.append(leaf_tensor)
                # print(f"🔍 Worker {worker_id}: Added tensor {i}, shape: {leaf_tensor.shape}")
            
            # 하나의 배치 메시지로 전송
            # print(f"🔍 Worker {worker_id}: Putting batch request in queue")
            req_q.put((worker_id, "batch", batch_tensors))
            
            # 하나의 배치 응답 받기
            # print(f"🔍 Worker {worker_id}: Waiting for batch response")
            batch_policies, batch_values = self._wait_for_batch_response("batch", resp_q, worker_id)
            # print(f"🔍 Worker {worker_id}: Got batch response: {len(batch_policies)} policies, {len(batch_values)} values")
            
            # 4. 결과 적용 및 역전파
            for i, (node, path) in enumerate(zip(all_leaf_nodes, all_leaf_paths)):
                if i < len(batch_policies):
                    policy = batch_policies[i]
                    value = batch_values[i].item()
                    
                    # 노드 확장
                    legal_moves = move_generator(node.position)
                    legal_policies = self._extract_legal_policies(legal_moves, policy, node.position.side)
                    node.expand(legal_moves, legal_policies)
                    
                    # 역전파
                    node.backpropagate(value)
                else:
                    # 응답이 부족한 경우 기본값으로 처리
                    print(f"🔍 Worker {worker_id}: Missing response for simulation {i}, using default")
                    node.backpropagate(0.0)
            
            # print(f"🔍 Worker {worker_id}: Processed {len(batch_policies)}/{batch_size} simulations")
        
        # 5. 결과 반환
        # print(f"🔍 Worker {worker_id}: Selecting best move")
        result = self._select_best_move(temperature, move_count)
        # print(f"🔍 Worker {worker_id}: Selected move: {result[0]}")
        return result
    
    def _extract_legal_policies(self, legal_moves, policy, side):
        """합법수에 대한 정책 확률 추출"""
        legal_policies = []
        for move in legal_moves:
            try:
                move_idx = move_to_policy_index(move, side)
                prob = policy[move_idx].item() if isinstance(policy, torch.Tensor) else policy[move_idx]
                legal_policies.append(prob)
            except:
                legal_policies.append(1.0 / len(legal_moves))
        
        # 정규화
        policy_sum = sum(legal_policies)
        if policy_sum > 0:
            legal_policies = [p / policy_sum for p in legal_policies]
        else:
            legal_policies = [1.0 / len(legal_policies) for _ in legal_policies]
        
        return legal_policies
    
    def _add_dirichlet_noise(self, policies, dirichlet_alpha):
        """디리클레 노이즈 추가"""
        alpha = dirichlet_alpha if dirichlet_alpha is not None else MCTS_DIRICHLET_ALPHA
        noise = np.random.dirichlet([alpha] * len(policies))
        noise_weight = MCTS_DIRICHLET_WEIGHT
        return (1 - noise_weight) * np.array(policies) + noise_weight * noise
    
    def _select_best_move(self, temperature, move_count):
        """최적 수 선택"""
        if temperature is None:
            temperature = MCTS_TEMPERATURE_EARLY if move_count < MCTS_TEMPERATURE_THRESHOLD else MCTS_TEMPERATURE_LATE
        
        if temperature == 0:
            best_move = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            moves = list(self.root.children.keys())
            visit_counts = [child.visit_count for child in self.root.children.values()]
            visit_counts_temp = [count**(1/temperature) for count in visit_counts]
            probs = [count / sum(visit_counts_temp) for count in visit_counts_temp]
            best_move = np.random.choice(moves, p=probs)
        
        # 정책 타겟 생성
        policy_target = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
        total_visits = sum(child.visit_count for child in self.root.children.values())
        
        if total_visits > 0:
            for move, child in self.root.children.items():
                move_idx = move_to_policy_index(move, self.root.position.side)
                policy_target[move_idx] = child.visit_count / total_visits
        
        return best_move, policy_target

    def _wait_for_batch_response(self, request_id, resp_q, worker_id):
        """
        배치 응답을 기다립니다.
        - "root": 단일 응답 (policy, value)
        - "batch": 배치 응답 (policies_list, values_list)
        """
        while True:
            w_id, r_id, *response = resp_q.get()
            if w_id == worker_id and r_id == request_id:
                if request_id == "root":
                    # 단일 응답
                    return response[0], response[1]  # policy, value
                elif request_id == "batch":
                    # 배치 응답
                    return response[0], response[1]  # policies_list, values_list
            else:
                # 다른 요청 결과는 다시 큐에 넣음
                resp_q.put((w_id, r_id, *response))

    def _wait_for_response(self, request_id, resp_q, worker_id):
        """
        신경망 평가 응답을 기다립니다.
        """
        while True:
            w_id, r_id, policy, value = resp_q.get()
            if w_id == worker_id and r_id == request_id:
                return policy, value
            else:
                # 다른 요청 결과는 다시 큐에 넣음
                resp_q.put((w_id, r_id, policy, value))

