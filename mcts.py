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
    """MCTS 트리의 노드"""
    def __init__(self, parent, prior_p, position=None):
        self.parent = parent
        self.children = {}  # {move: Node}
        self.visit_count = 0
        self.total_action_value = 0.0
        self.mean_action_value = 0.0
        self.prior_p = prior_p
        self.position = position # 이 노드가 나타내는 게임 상태 (AlphaZero 논문 참조)
        
        # 메모리 효율성을 위한 __slots__ 사용 (선택적)
        # __slots__ = ['parent', 'children', 'visit_count', 'total_action_value', 
        #              'mean_action_value', 'prior_p', 'position']

    def select(self, c_puct, logger=None, depth=0):
        """UCT(PUCT) 점수가 가장 높은 자식 노드 선택 (순수 AlphaZero) - 최적화된 버전"""
        best_score = -float('inf')
        best_move = None
        best_child = None

        # 제곱근 미리 계산 (반복 계산 방지)
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        # 모든 유효한 자식 노드에 대해 UCT 점수 계산 (순수 AlphaZero 방식)
        for move, child in self.children.items():
            if child.position is None:
                continue
            
            # UCT 점수 계산 (AlphaZero PUCT 공식) - 최적화된 버전
            if child.visit_count == 0:
                # 미방문 노드: prior_p를 기준으로 정렬 (AlphaZero 방식)
                uct_score = c_puct * child.prior_p * sqrt_parent_visits
            else:
                # 방문된 노드: Q + U 계산 (나눗셈 최적화)
                u_value = c_puct * child.prior_p * sqrt_parent_visits / (1 + child.visit_count)
                uct_score = child.mean_action_value + u_value
            
            if uct_score > best_score:
                best_score = uct_score
                best_move = move
                best_child = child

        # 선택된 move와 child 반환
        if best_move is None:
            print(f"Warning: No valid child found in select()")
            return None, None
            
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
        리프 노드의 평가값(value)을 루트까지 역전파합니다.
        AlphaZero 논문 (Methods -> Search)에 따라 N(s,a)와 W(s,a)를 업데이트하고 Q(s,a)를 계산합니다.
        """
        self.visit_count += 1
        self.total_action_value += value
        self.mean_action_value = self.total_action_value / self.visit_count
        if self.parent:
            # AlphaZero는 상대방의 시점에서의 value로 변환 (-value)하여 역전파합니다.
            self.parent.backpropagate(-value)
    
    def is_fully_expanded(self):
        """
        노드가 완전히 확장되었는지 확인합니다.
        모든 합법수에 대해 자식 노드가 생성되었으면 True를 반환합니다.
        """
        if self.position is None:
            return True
        
        # 이미 확장된 경우 (자식 노드가 있는 경우)
        return len(self.children) > 0
    
    def is_leaf(self):
        """
        리프 노드인지 확인합니다.
        자식 노드가 없으면 리프 노드입니다.
        """
        return len(self.children) == 0

    def get_uct_score(self, c_puct):
        """
        PUCT(Polynomial Upper Confidence Trees) 점수 계산.
        select 메서드와 일관된 로직 사용
        """
        # 부모 노드가 없으면 (루트 노드) UCT 점수 계산 불가
        if self.parent is None:
            return self.mean_action_value if self.visit_count > 0 else 0.0
            
        sqrt_parent_visits = math.sqrt(self.parent.visit_count)
        
        if self.visit_count == 0:
            # 미방문 노드: prior_p를 기준으로 계산 (select와 동일)
            return c_puct * self.prior_p * sqrt_parent_visits
        else:
            # 방문된 노드: Q + U 계산 (select와 동일)
            q_value = self.mean_action_value
            u_value = c_puct * self.prior_p * sqrt_parent_visits / (1 + self.visit_count)
            return q_value + u_value

    def is_leaf(self):
        """노드가 리프 노드인지 확인합니다."""
        return len(self.children) == 0

class MCTS:
    """몬테카를로 트리 탐색 클래스"""
    def __init__(self, c_puct=None):
        self.c_puct = c_puct if c_puct is not None else MCTS_C_PUCT
        self.root = None
        self.last_position = None

    def update_with_move(self, move, position):
        """
        트리 재사용: 선택한 move의 자식 노드를 새로운 루트로 승격
        position: 현재 실제 게임 포지션 (동기화용)
        """
        if self.root is not None and move in self.root.children:
            new_root = self.root.children[move]
            new_root.parent = None
            self.root = new_root
        else:
            # 트리 불일치(예: 상대방 수, 서브트리 없음)면 새로 생성
            self.root = None
        self.last_position = position

    # (구버전 search 함수 완전 제거, 아래 최신 search만 유지)
    def search(self, position, move_generator, num_simulations, req_q, resp_q, worker_id, move_count=0, logger=None, temperature=1.0, add_dirichlet_noise=True, dirichlet_alpha=None, reuse_tree=True):
        # 트리 재사용: position.hash_key가 다르면 update_with_move를 자동 호출
        if reuse_tree and self.root is not None and self.root.position.hash_key != position.hash_key:
            # root의 자식 중 position.hash_key와 일치하는 노드가 있으면 트리 승격
            found = False
            for move, child in self.root.children.items():
                if hasattr(child, 'position') and child.position.hash_key == position.hash_key:
                    child.parent = None
                    self.root = child
                    found = True
                    break
            if not found:
                self.root = None
        need_root_eval = not reuse_tree or self.root is None

        batch_nodes = []
        batch_paths = []

        if need_root_eval:
            # 루트 평가 + 시뮬레이션 배치를 한 번에 묶어서 보냄
            root_tensor = position_to_tensor_cached(position)
            sim_count = 0
            total_sim = 0
            # 루트 노드 생성 (아직 prior_p, policy 없음)
            root_node = Node(parent=None, prior_p=0, position=position)
            batch_nodes.append(root_node)
            batch_paths.append([root_node])
            # 시뮬레이션 노드들 수집
            for _ in range(num_simulations - 1):
                node = root_node
                path = [node]
                depth = 0
                while not node.is_leaf():
                    move, node = node.select(self.c_puct, None, depth)
                    if move is None or node is None:
                        break
                    path.append(node)
                    depth += 1
                if node is not None and node.position is not None:
                    batch_nodes.append(node)
                    batch_paths.append(path)
                sim_count += 1
                total_sim += 1
            # 배치 텐서 생성
            batch_tensors = [position_to_tensor_cached(n.position) for n in batch_nodes]
            batch_tensor = torch.stack(batch_tensors)
            req_q.put((worker_id, 'batch', batch_tensor))
            _, _, batch_policies, batch_values = self._wait_for_batch_response(resp_q, worker_id)
            # 루트 확장 및 역전파
            policy = batch_policies[0]
            value = batch_values[0].item()
            legal_moves = move_generator(position)
            legal_move_policies = []
            for move in legal_moves:
                try:
                    move_idx = move_to_policy_index(move, position.side)
                    prob = policy[move_idx].item() if isinstance(policy, torch.Tensor) else policy[move_idx]
                    legal_move_policies.append(prob)
                except Exception as e:
                    legal_move_policies.append(1.0 / len(legal_moves))
            if len(legal_move_policies) > 0:
                policy_sum = sum(legal_move_policies)
                if policy_sum > 0:
                    legal_move_policies = [p / policy_sum for p in legal_move_policies]
                else:
                    legal_move_policies = [1.0 / len(legal_move_policies) for _ in legal_move_policies]
            if add_dirichlet_noise:
                alpha = dirichlet_alpha if dirichlet_alpha is not None else MCTS_DIRICHLET_ALPHA
                noise = np.random.dirichlet([alpha] * len(legal_move_policies))
                noise_weight = MCTS_DIRICHLET_WEIGHT
                legal_move_policies = (1 - noise_weight) * np.array(legal_move_policies) + noise_weight * noise
            root_node.expand(legal_moves, legal_move_policies)
            root_node.backpropagate(value)
            self.root = root_node
            # 나머지 시뮬레이션 노드들 확장 및 역전파
            for i in range(1, len(batch_nodes)):
                node = batch_nodes[i]
                path = batch_paths[i]
                policy = batch_policies[i]
                value = batch_values[i].item()
                legal_moves_at_leaf = move_generator(node.position)
                legal_move_policies = []
                for move in legal_moves_at_leaf:
                    try:
                        move_idx = move_to_policy_index(move, node.position.side)
                        prob = policy[move_idx].item() if isinstance(policy, torch.Tensor) else policy[move_idx]
                        legal_move_policies.append(prob)
                    except Exception as e:
                        legal_move_policies.append(1.0 / len(legal_moves_at_leaf))
                if len(legal_move_policies) > 0:
                    policy_sum = sum(legal_move_policies)
                    if policy_sum > 0:
                        legal_move_policies = [p / policy_sum for p in legal_move_policies]
                    else:
                        legal_move_policies = [1.0 / len(legal_move_policies) for _ in legal_move_policies]
                node.expand(legal_moves_at_leaf, legal_move_policies)
                node.backpropagate(value)
        else:
            # 기존 트리 재사용: 시뮬레이션 배치만
            batch_nodes = []
            batch_paths = []
            for _ in range(num_simulations):
                node = self.root
                path = [node]
                depth = 0
                while not node.is_leaf():
                    move, node = node.select(self.c_puct, None, depth)
                    if move is None or node is None:
                        break
                    path.append(node)
                    depth += 1
                if node is not None and node.position is not None:
                    batch_nodes.append(node)
                    batch_paths.append(path)
            if batch_nodes:
                batch_tensors = [position_to_tensor_cached(n.position) for n in batch_nodes]
                batch_tensor = torch.stack(batch_tensors)
                req_q.put((worker_id, 'batch', batch_tensor))
                _, _, batch_policies, batch_values = self._wait_for_batch_response(resp_q, worker_id)
                for i, (node, path) in enumerate(zip(batch_nodes, batch_paths)):
                    policy = batch_policies[i]
                    value = batch_values[i].item()
                    legal_moves_at_leaf = move_generator(node.position)
                    legal_move_policies = []
                    for move in legal_moves_at_leaf:
                        try:
                            move_idx = move_to_policy_index(move, node.position.side)
                            prob = policy[move_idx].item() if isinstance(policy, torch.Tensor) else policy[move_idx]
                            legal_move_policies.append(prob)
                        except Exception as e:
                            legal_move_policies.append(1.0 / len(legal_moves_at_leaf))
                    if len(legal_move_policies) > 0:
                        policy_sum = sum(legal_move_policies)
                        if policy_sum > 0:
                            legal_move_policies = [p / policy_sum for p in legal_move_policies]
                        else:
                            legal_move_policies = [1.0 / len(legal_move_policies) for _ in legal_move_policies]
                    node.expand(legal_moves_at_leaf, legal_move_policies)
                    node.backpropagate(value)

        # 탐색 완료 후 루트의 방문 횟수를 기반으로 다음 수 선택
        move_counts = []
        moves = []
        move_scores = []  # UCT 점수 기록을 위해 추가
        for move, child in self.root.children.items():
            moves.append(move)
            move_counts.append(child.visit_count)
            sqrt_parent_visits = math.sqrt(self.root.visit_count)
            if child.visit_count == 0:
                uct_score = self.c_puct * child.prior_p * sqrt_parent_visits
            else:
                q_value = child.mean_action_value
                u_value = self.c_puct * child.prior_p * sqrt_parent_visits / (1 + child.visit_count)
                uct_score = q_value + u_value
            move_scores.append(uct_score)
        if not moves:
            print(f"Warning: No moves available from root!")
            return None, torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
        if temperature == 0.0:
            best_move = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
            selected_idx = moves.index(best_move)
            selection_reason = "greedy (temp=0)"
        else:
            move_probs = np.array(move_counts, dtype=np.float64)
            move_probs = move_probs ** (1.0 / temperature)
            move_probs = move_probs / np.sum(move_probs)
            selected_idx = np.random.choice(len(moves), p=move_probs)
            best_move = moves[selected_idx]
            selection_reason = f"temperature-based (temp={temperature:.2f})"
        if temperature == 0.0:
            move_visit_pairs = list(zip(moves, move_counts))
            move_visit_pairs.sort(key=lambda x: x[1], reverse=True)
            top_moves = move_visit_pairs[:3]
            top_moves_str = ", ".join([f"{move}({visits})" for move, visits in top_moves])
            print(f"  MCTS worker {worker_id}: {num_simulations} sims, selected {selection_reason}")
            print(f"    Top moves: {top_moves_str}, selected: {best_move}({move_counts[selected_idx]})")
            visited_moves = [(move, count) for move, count in zip(moves, move_counts) if count > 0]
            if len(visited_moves) > 1:
                print(f"    Total visited moves: {len(visited_moves)}/{len(moves)}")
            else:
                print(f"    Warning: Only {len(visited_moves)} moves visited out of {len(moves)} legal moves!")
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
        간단한 배치 누적 방식 - 핵심 아이디어만 구현
        """
        # 1. 루트 노드 초기화
        root_tensor = position_to_tensor_cached(position)
        req_q.put((worker_id, 0, root_tensor))
        policy, value = self._wait_for_response(0, resp_q, worker_id)
        
        legal_moves = move_generator(position)
        self.root = Node(parent=None, prior_p=0, position=position)
        
        # 루트 확장
        legal_move_policies = self._extract_legal_policies(legal_moves, policy, position.side)
        if add_dirichlet_noise:
            legal_move_policies = self._add_dirichlet_noise(legal_move_policies, dirichlet_alpha)
        
        self.root.expand(legal_moves, legal_move_policies)
        self.root.backpropagate(value.item())
        
        # 2. 간단한 배치 수집
        leaf_nodes = []
        leaf_paths = []
        
        for i in range(num_simulations):
            # 리프 노드까지 탐색
            path = []
            current = self.root
            
            while not current.is_leaf():
                move, current = current.select(self.c_puct)
                if current is None:
                    break
                path.append(current)
            
            if current is not None and current.position is not None:
                leaf_nodes.append(current)
                leaf_paths.append(path)
        
        # 3. 배치 평가 (핵심!)
        if leaf_nodes:
            # 모든 리프 노드를 한 번에 평가
            batch_tensors = [position_to_tensor_cached(node.position) for node in leaf_nodes]
            batch_tensor = torch.stack(batch_tensors)
            
            # 한 번의 통신으로 모든 평가 완료
            req_q.put((worker_id, 'batch', batch_tensor))
            _, _, batch_policies, batch_values = self._wait_for_batch_response(resp_q, worker_id)
            
            # 4. 결과 적용
            for i, (node, path) in enumerate(zip(leaf_nodes, leaf_paths)):
                policy = batch_policies[i]
                value = batch_values[i].item()
                
                # 노드 확장
                legal_moves = move_generator(node.position)
                legal_policies = self._extract_legal_policies(legal_moves, policy, node.position.side)
                node.expand(legal_moves, legal_policies)
                
                # 역전파
                for node_in_path in reversed(path):
                    node_in_path.backpropagate(value)
        
        # 5. 결과 반환
        return self._select_best_move(temperature, move_count)
    
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

    def _wait_for_batch_response(self, resp_q, worker_id):
        """배치 응답을 기다립니다."""
        while True:
            w_id, r_id, policy, value = resp_q.get()
            if w_id == worker_id and r_id == 'batch':
                return w_id, r_id, policy, value
            else:
                # 다른 요청 결과는 다시 큐에 넣음
                resp_q.put((w_id, r_id, policy, value))

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

