import torch
from ..torch_device import resolve_device
from .operators import EmbbedingSimilarityOperator,CosineOperator,cosine
from functools import partial
import dask.bag as db
import warnings
from typing import Callable, Literal, Optional, Union, List

@torch.no_grad()
def emb_similarity_cpu(
    query: torch.Tensor, # shape: (n_q, dim), dtype: float32
    ref: torch.Tensor, # shape: (n_r, dim), dtype: float32
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine,
    chunk_size: int = 5120,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
) -> torch.Tensor: # shape: (n_q, n_r), dtype: float32
    
    # 自动选择设备
    output_device = output_device or work_device
    query = query.to(work_device)
    ref = ref.to(work_device)
    
    # 分块计算逻辑    
    results = []
    for q_chunk in query.split(chunk_size):
        chunk_results = []
        for r_chunk in ref.split(chunk_size):
            res = sim_operator(q_chunk, r_chunk).to(output_device)
            chunk_results.append(res)
        results.append(torch.cat(chunk_results, dim=1))
    return torch.cat(results).to(output_device)

@torch.no_grad()
def emb_similarity_cuda(
    query: torch.Tensor,
    ref: torch.Tensor,
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine,
    chunk_size: int = 5120,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    
    output_device = output_device or work_device
    torch.cuda.set_device(work_device)
    
    # 创建三阶段流
    h2d_stream = torch.cuda.Stream()  # 主机到设备传输流
    compute_stream = torch.cuda.Stream()  # 计算流
    d2h_stream = torch.cuda.Stream()  # 设备到主机传输流
    
    results = []
    ref_chunks = list(ref.split(chunk_size))
    
    # 预取第一个ref chunk
    current_ref = ref_chunks[0].to(work_device, non_blocking=True)
    
    for q_chunk in query.to(work_device).split(chunk_size):
        chunk_results = []
        next_ref_iter = iter(ref_chunks[1:] + [None])
        
        for i in range(len(ref_chunks)):
            # 流水线阶段1：预取下一个chunk
            with torch.cuda.stream(h2d_stream):
                next_ref = next(next_ref_iter, None)
                if next_ref is not None:
                    next_ref = next_ref.to(work_device, non_blocking=True)
            
            # 流水线阶段2：执行计算
            with torch.cuda.stream(compute_stream):
                sim = sim_operator(q_chunk, current_ref)
            
            # 流水线阶段3：传输结果
            with torch.cuda.stream(d2h_stream):
                sim_cpu = sim.to(output_device,non_blocking=True)
                chunk_results.append(sim_cpu)
            
            # 更新当前ref并同步流
            current_ref = next_ref if next_ref is not None else current_ref
            torch.cuda.current_stream().wait_stream(h2d_stream)
            torch.cuda.current_stream().wait_stream(d2h_stream)
        
        # 等待所有操作完成
        torch.cuda.synchronize()
        results.append(torch.cat(chunk_results, dim=1))
    
    return torch.cat(results).to(output_device)

def emb_similarity(
    query: torch.Tensor,
    ref: torch.Tensor,
    sim_operator: EmbbedingSimilarityOperator = CosineOperator,
    chunk_size: int = 5120,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    operator_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    
    """计算查询向量和参考向量之间的相似度矩阵。
    
    该函数自动根据设备类型选择CPU或GPU实现，支持分块计算以处理大规模数据。
    
    Args:
        query: 查询向量张量，形状为(n_query, dim)
        ref: 参考向量张量，形状为(n_ref, dim)
        sim_operator: 相似度计算算子，默认为余弦相似度
        chunk_size: 分块大小，用于内存优化，默认为5120
        work_device: 计算设备，可以是'auto'、'cpu'、'cuda'或torch.device对象
        output_device: 输出设备，可以是'auto'、'cpu'、'cuda'或torch.device对象
        operator_kwargs: 传递给相似度算子的额外参数
        
    Returns:
        torch.Tensor: 相似度矩阵，形状为(n_query, n_ref)
    """

    # 自动推断工作设备
    _work_device = resolve_device(work_device, query.device)
    # 自动推断输出设备
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)
    
    # 分发到具体实现
    if _work_device.type.startswith('cuda'):
        return emb_similarity_cuda(
            query, ref, operator, chunk_size,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return emb_similarity_cpu(
            query, ref, operator, chunk_size,
            work_device=_work_device,
            output_device=_output_device
        )
        
def emb_similarity_by_queue(
    query_queue: List[torch.Tensor],
    ref_queue: List[torch.Tensor],
    sim_operator: EmbbedingSimilarityOperator = CosineOperator,
    chunk_size: int = 5120,
    num_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    operator_kwargs: Optional[dict] = None,
) -> List[torch.Tensor]:
    
    """批量计算多组查询和参考向量之间的相似度矩阵。
    
    该函数使用多线程并行处理多个相似度计算任务，自动根据设备类型选择最佳实现。
    
    Args:
        query_queue: 查询向量张量列表，每个张量形状为(n_query, dim)
        ref_queue: 参考向量张量列表，每个张量形状为(n_ref, dim)
        sim_operator: 相似度计算算子，默认为余弦相似度
        chunk_size: 分块大小，用于内存优化，默认为5120
        num_workers: 并行工作线程数，默认为4
        work_device: 计算设备，可以是'auto'、'cpu'、'cuda'或torch.device对象
        output_device: 输出设备，可以是'auto'、'cpu'、'cuda'或torch.device对象
        operator_kwargs: 传递给相似度算子的额外参数
        
    Returns:
        List[torch.Tensor]: 相似度矩阵列表，每个矩阵形状为(n_query, n_ref)
    """
    
    # 合法性检查
    assert len(query_queue) == len(ref_queue)
    
    # 自动设备
    _work_device = resolve_device(work_device, query_queue[0].device)
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)
    
    # 构造工作函数
    if _work_device.type.startswith('cuda'):
        work_func = partial(
            emb_similarity_cuda,
            chunk_size=chunk_size,
            sim_operator=operator,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        work_func = partial(
            emb_similarity_cpu,
            chunk_size=chunk_size,
            sim_operator=operator,
            work_device=_work_device,
            output_device=_output_device
        )
    
    # 任务分发
    if len(query_queue) > 1:
        
        # dask分发
        queue_bag = db.from_sequence(zip(query_queue, ref_queue), npartitions=num_workers)
        queue_results = queue_bag.map(lambda x: work_func(x[0],x[1]))
        
        # 计算
        results = queue_results.compute(scheduler='threads', num_workers=num_workers)
        
    elif len(query_queue) == 1:
        
        results = [work_func(query_queue[0], ref_queue[0])]
        
    else:
        
        warnings.warn("Empty query queue, return empty result.")
        results = []
    
    return results