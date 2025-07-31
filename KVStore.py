import os
import hashlib
from collections import OrderedDict
from typing import Dict, Tuple, Optional
from typing import List, Optional, Tuple, Union
import torch
from transformers.cache_utils import Cache
from copy import deepcopy
import threading
from time import time
import multiprocessing
from collections import deque
import copy
from transformers import AutoTokenizer
import random

from cache_utils import DynamicCache
Digest = Tuple[torch.Tensor, torch.Tensor]

def create_cache_copy(existing_cache: DynamicCache) -> DynamicCache:
    # 创建一个新的 DynamicCache 实例
    cache_copy = DynamicCache()
        
    # 手动复制现有缓存的属性
    cache_copy._seen_tokens = existing_cache._seen_tokens
    cache_copy.key_cache = [tensor.clone() for tensor in existing_cache.key_cache]  # 使用 clone() 来确保副本独立
    cache_copy.value_cache = [tensor.clone() for tensor in existing_cache.value_cache]  # 使用 clone() 来确保副本独立

    return cache_copy

class KVCacheManager:
    def __init__(self, 
                 memory_capacity: int = 2000, 
                 hbm_capacity = 20,
                 disk_path: Optional[str] = "/mnt/sda1/homie_cache/",
                 block_size: int = 64,
                 topk_threshold: float = 0.9):
        self.disk_path = disk_path
        self.hbm_capacity = hbm_capacity
        self.memory_capacity = memory_capacity
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/sda1/Llama-3.1-8B")
        
        self.history_token_cnt = 0
        self.input_ids_list: List[List[int]] = []
        self.hash_key_list: List[str] = []
        self.kv_cache: Dict[str, DynamicCache] = {}
        self.memory_digest_cache: Dict[str, List[Digest]] = {}
        self.retrieve_cnt = -1
        
        # Logging setup
        self.log_dir = f"logs/block_size_{block_size}_threshold_{topk_threshold}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.topk_log_file = os.path.join(self.log_dir, "dynamic_topk.log")
        self.time_log_file = os.path.join(self.log_dir, "timing.log")

        # Clear log files at initialization
        with open(self.topk_log_file, "w") as f:
            f.write("request_id,layer_idx,max_k\n")
        with open(self.time_log_file, "w") as f:
            f.write("request_id,total_forward_time,total_topk_time,total_concat_time,total_attn_time,other_time\n")

        # Time accumulators for the current request
        self.current_topk_time = 0
        self.current_concat_time = 0
        self.current_attn_time = 0
        
    def _generate_key(self, token_ids: List[int]) -> str:
        return hashlib.sha256(str(token_ids).encode()).hexdigest()
    
    def get_cache(self, 
                 token_ids: List[int]
                ) -> Optional[Cache]:
        key = self._generate_key(token_ids)
        if key in self.kv_cache:
            return self.kv_cache[key]
        else:
            return None
        
    def is_exist(self, 
                 token_ids: List[int]
                ) -> bool:
        key = self._generate_key(token_ids)
        return key in self.kv_cache
        
    def save_cache(self, token_ids: List[int], cache: DynamicCache):
        self.input_ids_list.append(token_ids)
        key = self._generate_key(token_ids)
        self.hash_key_list.append(key)
        copy_cache = DynamicCache()
        for i in range(len(cache.key_cache)):
            cpu_k_tensor = cache.key_cache[i].cpu()
            cpu_v_tensor = cache.value_cache[i].cpu()
            copy_cache.key_cache.append(copy.deepcopy(cpu_k_tensor))
            copy_cache.value_cache.append(copy.deepcopy(cpu_v_tensor))
        self.kv_cache[key] = copy_cache

    def save_digest_cache(self, token_ids: List[int], key_cache: List[torch.Tensor]):
        # (bsz, head_num, k_len, head_dim)
        key = self._generate_key(token_ids)
        digest = []

        # Bounding-cuboid digest
        for key_tensor in key_cache:
            # Clone the key_tensor to ensure the original key_cache is not modified
            key_tensor_clone = key_tensor.clone()
            maxs = key_tensor_clone.max(dim=2).values
            mins = key_tensor_clone.min(dim=2).values
            centers = (maxs + mins) / 2
            dists = (
                (centers.unsqueeze(2) - key_tensor_clone).abs().mean(dim=2)
            )
            maxs = centers + dists
            mins = centers - dists
            digest.append((maxs, mins))

        self.memory_digest_cache[key] = digest

    def log_time(self, event_type: str, duration: float):
        if event_type == "topk_time":
            self.current_topk_time += duration
        elif event_type == "concat_time":
            self.current_concat_time += duration
        elif event_type == "attn_time":
            self.current_attn_time += duration

    def finalize_and_log_times(self, total_forward_time: float):
        other_time = total_forward_time - self.current_topk_time - self.current_concat_time - self.current_attn_time
        with open(self.time_log_file, "a") as f:
            f.write(f"{self.retrieve_cnt},{total_forward_time},{self.current_topk_time},{self.current_concat_time},{self.current_attn_time},{other_time}\n")
        
        # Reset accumulators
        self.current_topk_time = 0
        self.current_concat_time = 0
        self.current_attn_time = 0
        
    def retrieve_related_kv(self,
                            query_vector: torch.Tensor,
                            topk_threshold: float = 0.9,
                            layer_idx: int = 0,
                            num_key_value_groups: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self.retrieve_cnt += 1

        if len(self.hash_key_list) == 0 or topk_threshold == 0:
            return torch.empty(0, 0, 0, 0, device=query_vector.device), torch.empty(0, 0, 0, 0, device=query_vector.device)

        # TODO: 这里的tensor构建是十分低效的
        max_digest_list = []
        min_digest_list = []
        for key in self.hash_key_list:
            if len(self.memory_digest_cache[key]) > 0:
                max_digest_list.append(self.memory_digest_cache[key][layer_idx][0])
                min_digest_list.append(self.memory_digest_cache[key][layer_idx][1])
        
        if not max_digest_list:
            return torch.empty(0, 0, 0, 0, device=query_vector.device), torch.empty(0, 0, 0, 0, device=query_vector.device)

        max_digest = torch.stack(max_digest_list, dim=2).to(query_vector.device)
        min_digest = torch.stack(min_digest_list, dim=2).to(query_vector.device)
        
        st = time()
        scores = self.related_score_eval(query_vector, max_digest, min_digest, layer_idx, num_key_value_groups)
        
        head_num, k_len = scores.shape
        
        # Sort scores and get original indices
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        
        # Calculate cumulative scores
        cumulative_scores = torch.cumsum(sorted_scores, dim=-1)
        
        # Calculate total score for each head
        total_scores = cumulative_scores[:, -1].unsqueeze(-1)
        
        # Determine the number of top-k to select for each head
        # Find the first index where cumulative score exceeds the threshold
        k_for_threshold = (cumulative_scores > total_scores * topk_threshold).int().argmax(dim=-1) + 1
        
        # Use the maximum k across all heads to ensure uniform k_len
        max_k = k_for_threshold.max().item()
        self.log_time("topk_time", time() - st)
        
        # Log the dynamic topk value
        with open(self.topk_log_file, "a") as f:
            f.write(f"{self.retrieve_cnt},{layer_idx},{max_k}\n")
        print(f"Request: {self.retrieve_cnt}, Layer: {layer_idx}, Dynamic TopK: {max_k}")

        # Get the top indices based on max_k
        top_indices = sorted_indices[:, :max_k]

        st = time()
        head_retrieve_k_list = []
        head_retrieve_v_list = []
        for head_idx in range(head_num):
            head_top_indices = top_indices[head_idx]
            
            # Sort these indices to maintain order for concatenation
            sorted_head_top_indices = torch.sort(head_top_indices).values
            
            k_list = []
            v_list = []
            for idx in sorted_head_top_indices.tolist():
                key = self.hash_key_list[idx]
                k_list.append(self.kv_cache[key].key_cache[layer_idx][:, head_idx:head_idx+1, :, :])
                v_list.append(self.kv_cache[key].value_cache[layer_idx][:, head_idx:head_idx+1, :, :])
            
            if k_list:
                head_retrieve_k_list.append(torch.cat(k_list, dim=-2).cuda())
                head_retrieve_v_list.append(torch.cat(v_list, dim=-2).cuda())

        if not head_retrieve_k_list:
            return torch.empty(0, 0, 0, 0, device=query_vector.device), torch.empty(0, 0, 0, 0, device=query_vector.device)

        retrieve_k = torch.cat(head_retrieve_k_list, dim=1).cuda()
        retrieve_v = torch.cat(head_retrieve_v_list, dim=1).cuda()
        self.log_time("concat_time", time() - st)
        
        return retrieve_k, retrieve_v
        
    def related_score_eval(self, query: torch.Tensor, max_digest: torch.Tensor, min_digest: torch.Tensor,
                           layer_idx: int = 0, num_key_value_groups: int = 8) -> torch.Tensor:
        batch, q_num_key_value_heads, qlen, head_dim = query.shape
        batch, k_num_key_value_heads, klen, head_dim = max_digest.shape
        # GQA 适配
        n_rep = q_num_key_value_heads // k_num_key_value_heads
        max_digest = max_digest[:, :, None, :, :].expand(batch, k_num_key_value_heads, n_rep, klen, head_dim)
        max_digest = max_digest.reshape(batch, k_num_key_value_heads * n_rep, klen, head_dim)
        min_digest = min_digest[:, :, None, :, :].expand(batch, k_num_key_value_heads, n_rep, klen, head_dim)
        min_digest = min_digest.reshape(batch, k_num_key_value_heads * n_rep, klen, head_dim)

        # 广播计算
        max_digest_expand = max_digest.unsqueeze(2)
        min_digest_expand = min_digest.unsqueeze(2)
        query_expand = query.unsqueeze(3)
        qmax = query_expand * max_digest_expand
        qmin = query_expand * min_digest_expand  # (bsz, head_num, q_len, k_len, head_dim)

        detailed_scores = torch.max(qmax, qmin).sum(dim=-1) # (bsz, head_num, q_len, k_len)
        #detailed_scores = torch.relu(detailed_scores)
        
        # 保存所有层的详细分数
        #scores_to_save = detailed_scores.cpu()
        #save_path = os.path.join(self.score_dir, f"cnt_{self.retrieve_cnt}_layer_{layer_idx}_related_score.pt")
        #torch.save(scores_to_save, save_path)
    
        # 检查是否有 inf 或 NaN
        if torch.isinf(detailed_scores).any() or torch.isnan(detailed_scores).any():
            print("scores contains inf or NaN")
            assert 1==2
        RRF = True
        if RRF:
            # 将 scores 转换为排名
            ranks = torch.argsort(detailed_scores, dim=-1, descending=True).argsort(dim=-1) + 1  # (bsz, head_num, q_len, k_len)
            # 应用 RRF 公式
            k = 60  # RRF 参数，通常设置为 60
            rrf_scores = 1 / (k + ranks.float())  # (bsz, head_num, q_len, k_len)
            related_score = rrf_scores.sum(dim=-2)  # (bsz, head_num, k_len)
        else:
            related_score = detailed_scores.sum(dim=-2) # (bsz, head_num, k_len)
        
        related_score = related_score.reshape(batch, k_num_key_value_heads, n_rep, klen)
        # 对每个组求和
        related_score = related_score.sum(dim=2)  # (bsz, q_num_key_value_heads, k_len)
        return related_score.squeeze(0) # (head_num, k_len)
