import torch
from ..torch_device import resolve_device
from ..similarity.ms_similarity import (
    ms_entropy_similarity,
    SpectramSimilarityOperator,
    MSEntropyOperator,
)
import dask
import dask.bag as db
import pandas as pd
from typing import Tuple,Callable,Optional,Union,Literal,List

@torch.no_grad()
def spec_similarity_search_cpu(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    top_k: Optional[int] = None,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    output_device = output_device or work_device
    top_k = top_k or len(ref)
    
    # 缓冲区模板
    scores_template = torch.full((top_k,), -float('inf'), 
                                device=work_device, dtype=torch.float32)
    indices_template = torch.full((top_k,), -1, 
                                device=work_device, dtype=torch.long)
    
    # 单query搜索闭包
    def _search_single_query(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 初始化缓冲区
        scores_buf = scores_template.clone()
        indices_buf = indices_template.clone()
        current_count = 0  # 有效结果计数器
        
        q_tensor = q.to(work_device)
        
        for r_idx, r_spec in enumerate(ref):
            score = sim_operator(q_tensor, r_spec.to(work_device))
            
            # 仅处理可能进入TopK的情况
            if score > scores_buf.min() or current_count < top_k:
                # 合并到临时缓冲区
                temp_scores = torch.cat([scores_buf[:current_count], score.unsqueeze(0)])
                temp_indices = torch.cat([indices_buf[:current_count], 
                                        torch.tensor([r_idx], device=work_device)])
                
                # 获取排序后的索引
                sorted_idx = torch.argsort(temp_scores, descending=True)
                
                # 更新主缓冲区
                keep = min(top_k, len(temp_scores))
                scores_buf[:keep] = temp_scores[sorted_idx][:keep]
                indices_buf[:keep] = temp_indices[sorted_idx][:keep]
                current_count = min(current_count + 1, top_k)

        return scores_buf.to(output_device), indices_buf.to(output_device)

    # Dask并行处理
    query_bag = db.from_sequence(query, npartitions=num_dask_workers)
    query_bag = query_bag.map(_search_single_query)
    results = query_bag.compute(scheduler=dask_mode,num_workers=num_dask_workers)
    
    # 堆叠结果
    results = pd.DataFrame(results,columns=["scores", "indices"])
    scores = torch.stack(results['scores'].tolist())
    indices = torch.stack(results['indices'].tolist())
    
    return scores, indices

@torch.no_grad()
def spec_similarity_search_cpu_by_queue(
    query: List[List[torch.Tensor]],  # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    top_k: Optional[int] = None,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:

    output_device = output_device or work_device
    top_k = top_k or len(ref)
    
    # 缓冲区模板
    scores_template = torch.full((top_k,), -float('inf'), 
                                device=work_device, dtype=torch.float32)
    indices_template = torch.full((top_k,), -1, 
                                device=work_device, dtype=torch.long)
    
    # 单query搜索闭包
    def _search_single_query(i: int, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 初始化缓冲区
        scores_buf = scores_template.clone()
        indices_buf = indices_template.clone()
        current_count = 0  # 有效结果计数器
        
        q_tensor = q.to(work_device)
        
        for r_idx, r_spec in enumerate(ref[i]):
            score = sim_operator(q_tensor, r_spec.to(work_device))
            
            # 仅处理可能进入TopK的情况
            if score > scores_buf.min() or current_count < top_k:
                # 合并到临时缓冲区
                temp_scores = torch.cat([scores_buf[:current_count], score.unsqueeze(0)])
                temp_indices = torch.cat([indices_buf[:current_count], 
                                        torch.tensor([r_idx], device=work_device)])
                
                # 获取排序后的索引
                sorted_idx = torch.argsort(temp_scores, descending=True)
                
                # 更新主缓冲区
                keep = min(top_k, len(temp_scores))
                scores_buf[:keep] = temp_scores[sorted_idx][:keep]
                indices_buf[:keep] = temp_indices[sorted_idx][:keep]
                current_count = min(current_count + 1, top_k)

        return scores_buf.to(output_device), indices_buf.to(output_device)
    
    # 构建配对序列
    bag_queue = []
    for i,query_block in enumerate(query):
        query_block_bag = db.from_sequence(zip([i]*len(query_block), query_block), npartitions=num_dask_workers)
        results_bag = query_block_bag.map(lambda x: _search_single_query(x[0], x[1]))
        bag_queue.append(results_bag)
    
    # 并行搜索
    queue_results = dask.compute(bag_queue, scheduler=dask_mode, num_workers=num_dask_workers)[0]
    
    # 合并结果
    queue_results_bag = db.from_sequence(queue_results, npartitions=num_dask_workers)
    queue_results_bag = queue_results_bag.map(lambda x: pd.DataFrame(x,columns=["scores", "indices"]))
    queue_results_bag = queue_results_bag.map(lambda x: (torch.stack(x['scores'].tolist()),torch.stack(x['indices'].tolist())))
    queue_results = queue_results_bag.compute(scheduler=dask_mode, num_workers=num_dask_workers)
    
    return queue_results

@torch.no_grad()
def spec_similarity_search_cuda(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    top_k: Optional[int] = None,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

@torch.no_grad()
def spec_similarity_search_cuda_by_queue(
    query: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pass

def spec_similarity_search(
    query: List[torch.Tensor],
    ref: List[torch.Tensor],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

def spec_similarity_search_by_queue(
    query: List[List[torch.Tensor]],
    ref: List[List[torch.Tensor]],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pass