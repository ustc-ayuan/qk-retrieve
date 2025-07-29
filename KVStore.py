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
                 disk_path: Optional[str] = "/mnt/sda1/homie_cache/"):
        self.disk_path = disk_path
        self.hbm_capacity = hbm_capacity
        self.memory_capacity = memory_capacity
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/sda1/Meta-Llama-3-8B-Instruct")
        
        self.history_token_cnt = 0
        self.input_ids_list: List[List[int]] = []
        self.hash_key_list: List[str] = []
        self.kv_cache: Dict[str, DynamicCache] = {}
        self.memory_digest_cache: Dict[str, List[Digest]] = {}
        #self.top_indices_log_file = "top_indices.log"
        # Clear the log file at initialization
        #with open(self.top_indices_log_file, "w") as f:
        #    pass
        
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
        

    def retrieve_related_kv(self,
                            query_vector: torch.Tensor,
                            topk: int = 5,
                            layer_idx: int = 0) -> List[Tuple[int, Cache, str]]:
        if len(self.hash_key_list) == 0:
            return []
        
        # TODO: 这里的tensor构建是十分低效的
        max_digest_list = []
        min_digest_list = []
        for key in self.hash_key_list:
            if len(self.memory_digest_cache[key]) > 0:
                max_digest_list.append(self.memory_digest_cache[key][layer_idx][0])
                min_digest_list.append(self.memory_digest_cache[key][layer_idx][1])
        max_digest = torch.stack(max_digest_list, dim=2).to(query_vector.device)
        min_digest = torch.stack(min_digest_list, dim=2).to(query_vector.device)
        scores = self.related_score_eval(query_vector,max_digest,min_digest)
        topk = min(topk, len(scores[0]))
        top_values, top_indices = torch.topk(torch.tensor(scores[0]), k=topk, largest=True, sorted=True)
        #with open(self.top_indices_log_file, "a") as f:
        #    f.write(" ".join(map(str, top_indices.tolist())) + "\n")
        sorted_indices = torch.argsort(top_indices)
        results = []
        for sorted_idx in sorted_indices.tolist():
            idx = top_indices[sorted_idx].item()
            key = self.hash_key_list[idx]
            text = self.tokenizer.decode(self.input_ids_list[idx], skip_special_tokens=True)
            results.append((idx, self.kv_cache[key], text))
        """
        if layer_idx == 0:
            print("=============================================================================")
            #print("+ topk_scores: " + str(scores))
            print("layer_idx:",layer_idx)
            print("top_values",top_values)
            print("top_indices",top_indices)
            #for i in range(len(results)):
            #    print("+ The "+str(i+1)+"-idx is: " +str(results[i][0]))
            #    print("+ The "+str(i+1)+"-text is: \" " +results[i][2] +"\"")
            print("=============================================================================")
        #&"""
        return results
        
    def related_score_eval(self, query: torch.Tensor, max_digest: torch.Tensor, min_digest: torch.Tensor) -> List[float]:
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

        scores = torch.max(qmax, qmin).sum(dim=-1).sum(dim=1)  # (bsz, q_len, k_len)
    
        # 检查是否有 inf 或 NaN
        if torch.isinf(scores).any() or torch.isnan(scores).any():
            print("scores contains inf or NaN")
            assert 1==2
        RRF = False
        if RRF:
            # 将 scores 转换为排名
            ranks = torch.argsort(scores, dim=-1, descending=True).argsort(dim=-1) + 1  # (bsz, q_len, k_len)

            # 应用 RRF 公式
            k = 60  # RRF 参数，通常设置为 60
            rrf_scores = 1 / (k + ranks.float())  # (bsz, q_len, k_len)
            related_score = rrf_scores.sum(dim=-2)  # (bsz, k_len)
        else:
            scores = scores - scores.max(dim=-1, keepdim=True).values
            scores = scores / scores.abs().max(dim=-1, keepdim=True).values
            scores = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            related_score = scores.sum(dim=-2).tolist()
        return related_score
