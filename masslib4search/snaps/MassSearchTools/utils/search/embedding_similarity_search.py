import torch
from ..torch_device import resolve_device
from ..results_tools import topk_to_hit
from ..similarity.embedding_similarity import (
    cosine,
    EmbbedingSimilarityOperator,
    CosineOperator,
)
import dask.bag as db
from functools import partial
from typing import Tuple,Callable,Optional,Union,Literal,List
    
@torch.no_grad()
def emb_similarity_search_cpu(
    query: torch.Tensor, # shape: (n_q, dim), dtype: float32
    ref: torch.Tensor, # shape: (n_r, dim), dtype: float32
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine,
    top_k: Optional[int] = None,
    chunk_size: int = 5120,
    output_mode: Literal['top_k', 'hit'] = 'top_k',
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 设备配置
    output_device = output_device or work_device
    ref_num = ref.size(0)  # 获取参考集总数
    top_k = ref_num if top_k is None else top_k  # 自动对齐参考集数量
    
    # 空查询集
    if len(query) == 0:
        return (
            torch.tensor([], device=output_device, dtype=torch.long).reshape(0, top_k),
            torch.tensor([], device=output_device, dtype=torch.float32).reshape(0, top_k)
        )
        
    # 空参考集
    if len(ref) == 0:
        return (
            torch.full((len(query),top_k), -1, device=output_device, dtype=torch.long),
            torch.full((len(query),top_k), -float('inf'), device=output_device, dtype=torch.float32),
        )
    
    # 初始化全局缓冲区模板
    scores_template = torch.full((top_k,), -float('inf'), 
                                device=work_device, dtype=torch.float32)
    indices_template = torch.full((top_k,), -1, 
                                device=work_device, dtype=torch.long)
    
    results = []
    indices_list = []

    # 分块处理查询集
    for q_chunk in query.split(chunk_size):
        q_work = q_chunk.to(work_device)
        batch_size = q_work.size(0)
        
        # 初始化每批查询的缓冲区 (batch_size, top_k)
        scores_buf = scores_template[None, :].expand(batch_size, -1).clone()
        indices_buf = indices_template[None, :].expand(batch_size, -1).clone()

        # 分块处理参考集
        for r_idx, r_chunk in enumerate(ref.split(chunk_size)):
            r_work = r_chunk.to(work_device)
            sim = sim_operator(q_work, r_work)  # (batch_size, ref_chunk_size)
            
            # 生成全局索引
            start_idx = r_idx * chunk_size
            indices = torch.arange(start_idx, start_idx + r_work.size(0), 
                                    device=work_device)
            
            # 向量化合并逻辑
            combined_scores = torch.cat([scores_buf, sim], dim=1)
            combined_indices = torch.cat([
                indices_buf, 
                indices.expand(batch_size, -1)
            ], dim=1)
            
            # 保留TopK
            top_scores, top_pos = torch.topk(combined_scores, top_k, dim=1)
            scores_buf = top_scores
            indices_buf = torch.gather(combined_indices, 1, top_pos)

        # 后处理：确保严格排序（仅在需要时）
        if top_k < ref_num:
            sorted_idx = torch.argsort(scores_buf, dim=1, descending=True)
            scores_buf = torch.gather(scores_buf, 1, sorted_idx)
            indices_buf = torch.gather(indices_buf, 1, sorted_idx)
        
        # 转移结果到目标设备
        results.append(scores_buf.to(output_device))
        indices_list.append(indices_buf.to(output_device))

    I, S = torch.cat(indices_list, dim=0), torch.cat(results, dim=0)
    
    # 切换输出格式
    if output_mode == 'hit':
        I, S = topk_to_hit(I, S)
    
    return I, S

@torch.no_grad()
def emb_similarity_search_cuda(
    query: torch.Tensor, # shape: (n_q, dim), dtype: float32
    ref: torch.Tensor, # shape: (n_r, dim), dtype: float32
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine,
    top_k: Optional[int] = None,
    chunk_size: int = 5120,
    output_mode: Literal['top_k', 'hit'] = 'top_k',
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 设备配置
    output_device = output_device or work_device
    is_same_device = work_device == output_device
    ref_num = ref.size(0)
    top_k = ref_num if top_k is None else top_k  # 自动对齐参考集数量
    
    # 空查询集
    if len(query) == 0:
        return (
            torch.tensor([], device=output_device, dtype=torch.long).reshape(0, top_k),
            torch.tensor([], device=output_device, dtype=torch.float32).reshape(0, top_k)
        )
        
    # 空参考集
    if len(ref) == 0:
        return (
            torch.full((len(query),top_k), -1, device=output_device, dtype=torch.long),
            torch.full((len(query),top_k), -float('inf'), device=output_device, dtype=torch.float32),
        )
    
    # 初始化三流
    input_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    output_stream = torch.cuda.Stream()

    # 数据预处理（异步）
    with torch.cuda.stream(input_stream):
        query = query.to(work_device, non_blocking=True)
        ref = ref.to(work_device, non_blocking=True)
    torch.cuda.synchronize()

    # 缓冲区模板（固定内存）
    with torch.cuda.stream(compute_stream):
        scores_template = torch.full((top_k,), -float('inf'), 
                                    device=work_device, dtype=torch.float32)
        indices_template = torch.full((top_k,), -1, 
                                    device=work_device, dtype=torch.long)

    all_scores, all_indices = [], []
    r_chunks = list(ref.split(chunk_size))

    # 主处理循环
    for q_chunk in query.split(chunk_size):
        # 初始化缓冲区（每个查询块独立）
        with torch.cuda.stream(compute_stream):
            batch_size = q_chunk.size(0)
            scores_buf = scores_template[None, :].expand(batch_size, -1).clone()
            indices_buf = indices_template[None, :].expand(batch_size, -1).clone()

        # 预加载第一个参考块
        current_r = None
        with torch.cuda.stream(input_stream):
            current_r = r_chunks[0].to(work_device, non_blocking=True)
        compute_events = [torch.cuda.Event() for _ in r_chunks]

        for i in range(len(r_chunks)):
            # 预加载下一个参考块
            next_r = r_chunks[i+1].to(work_device, non_blocking=True) if i+1 < len(r_chunks) else None
            with torch.cuda.stream(input_stream):
                if next_r is not None:
                    next_r = next_r.to(work_device, non_blocking=True)

            # 计算块处理
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_stream(input_stream)
                
                # 计算相似度
                sim = sim_operator(q_chunk, current_r)
                indices = torch.arange(
                    i * chunk_size, 
                    i * chunk_size + current_r.size(0), 
                    device=work_device
                )

                # 合并到缓冲区
                combined_scores = torch.cat([scores_buf, sim], dim=1)
                combined_indices = torch.cat([
                    indices_buf,
                    indices[None, :].expand(batch_size, -1)
                ], dim=1)
                
                # 保留TopK
                scores_buf, top_pos = torch.topk(combined_scores, top_k, dim=1)
                indices_buf = torch.gather(combined_indices, 1, top_pos)
                
                compute_events[i].record()

            current_r = next_r

        # 最终排序（非全量模式）
        with torch.cuda.stream(compute_stream):
            if top_k < ref_num:
                sorted_idx = torch.argsort(scores_buf, dim=1, descending=True)
                scores_buf = torch.gather(scores_buf, 1, sorted_idx)
                indices_buf = torch.gather(indices_buf, 1, sorted_idx)

        # 异步传输结果
        with torch.cuda.stream(output_stream):
            compute_events[-1].synchronize()
            transfer = (scores_buf if is_same_device 
                        else scores_buf.to(output_device, non_blocking=True))
            all_scores.append(transfer)
            transfer = (indices_buf if is_same_device 
                        else indices_buf.to(output_device, non_blocking=True))
            all_indices.append(transfer)

    # 全局同步
    torch.cuda.synchronize()
    I, S = torch.cat(all_indices, dim=0),torch.cat(all_scores, dim=0)
    
    # 切换输出格式
    if output_mode == 'hit':
        I, S = topk_to_hit(I, S)
        
    return I, S

def emb_similarity_search(
    query: torch.Tensor,
    ref: torch.Tensor,
    sim_operator: EmbbedingSimilarityOperator = CosineOperator,
    top_k: int = None,
    chunk_size: int = 5120,
    output_mode: Literal['top_k', 'hit'] = 'top_k',
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    operator_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    计算查询向量集与参考向量集之间的相似度，并根据相似度返回每个查询向量前top_k个最相关的参考向量的索引和相似度分数。
    
    参数:
    - query: 查询向量集，形状为(n_q, dim)，其中n_q是查询向量的数量，dim是向量的维度。
    - ref: 参考向量集，形状为(n_r, dim)，其中n_r是参考向量的数量，dim是向量的维度。
    - sim_operator: 用于计算相似度的算子，默认为余弦相似度算子(CosineOperator)。
    - top_k: 每个查询向量返回前top_k个最相关的参考向量的索引和相似度。如果不指定，则返回与所有参考向量的相似度。
    - chunk_size: 用于分块处理的大小，以减少内存使用。
    - output_mode: 返回结果的格式
        - 'top_k'：返回每个查询向量的top_k个参考向量索引和相似度。
        - 'hit'：以命中对的形式返回平整的索引和相似度。
    - work_device: 在其上执行计算的设备，默认为'auto'，表示自动推断设备。
    - output_device: 在其上返回结果的设备，默认为'auto'，表示自动推断设备。
    - operator_kwargs: 传递给sim_operator的额外参数，可选。
    
    返回:
    - 如果output_mode为'top_k':
        - indices: 形状为(n_q, top_k)的参考向量索引矩阵，数据类型为long。包含每个查询向量最相关的top_k个参考向量的索引，无效索引用-1表示。
        - scores: 形状为(n_q, top_k)的相似度分数矩阵，数据类型为float。包含每个查询向量与最相关的top_k个参考向量的相似度分数，无效分数用-inf表示。
    - 如果output_mode为'hit':
        - new_indices: 形状为(num_hitted, 3)的矩阵，每行格式为[qry_index, ref_index, top_k_pos]，数据类型为long。
        - new_scores: 形状为(num_hitted,)的张量，包含有效命中的相似度分数，数据类型为float。
    - 所有返回结果均按照scores进行排序，从高到低。
    """
    
    # 自动推断工作设备
    _work_device = resolve_device(work_device, query.device)
    # 自动推断输出设备
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)
    
    if query.device.type.startswith("cuda"):
        return emb_similarity_search_cuda(
            query, ref, operator, top_k, chunk_size, output_mode, _work_device, _output_device
        )
    else:
        return emb_similarity_search_cpu(
            query, ref, operator, top_k, chunk_size, output_mode, _work_device, _output_device
        )
        
def emb_similarity_search_by_queue(
    query_queue: List[torch.Tensor],
    ref_queue: List[torch.Tensor],
    sim_operator: EmbbedingSimilarityOperator = CosineOperator,
    top_k: int = None,
    chunk_size: int = 5120,
    output_mode: Literal['top_k', 'hit'] = 'top_k',
    num_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    operator_kwargs: Optional[dict] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    
    """
    使用队列中的多个查询向量集和参考向量集，计算它们之间的相似度，并返回每个查询向量集与参考向量集之间前top_k个最相关的参考向量的索引和相似度分数。
    
    参数:
    - query_queue: 查询向量集的列表，每个查询向量集的形状为(n_q, dim)。
    - ref_queue: 参考向量集的列表，每个参考向量集的形状为(n_r, dim)。
    - sim_operator: 用于计算相似度的算子，默认为余弦相似度算子(CosineOperator)。
    - top_k: 每个查询向量返回前top_k个最相关的参考向量的索引和相似度。如果不指定，则返回与所有参考向量的相似度。
    - chunk_size: 用于分块处理的大小，以减少内存使用。
    - output_mode: 返回结果的格式
        - 'top_k'：返回每个查询向量的top_k个参考向量索引和相似度。
        - 'hit'：以命中对的形式返回平整的索引和相似度。
    - num_workers: 用于处理任务的线程数，用于并行处理。
    - work_device: 在其上执行计算的设备，默认为'auto'，表示自动推断设备。
    - output_device: 在其上返回结果的设备，默认为'auto'，表示自动推断设备。
    - operator_kwargs: 传递给sim_operator的额外参数，可选。
    
    返回:
    - results: 一个包含多个元组的列表，每个元组对应于query_queue和ref_queue中的一个查询向量集和参考向量集对。
        - 如果output_mode为'top_k':
            - indices: 形状为(n_q, top_k)的参考向量索引矩阵，数据类型为long。包含每个查询向量最相关的top_k个参考向量的索引，无效索引用-1表示。
            - scores: 形状为(n_q, top_k)的相似度分数矩阵，数据类型为float。包含每个查询向量与最相关的top_k个参考向量的相似度分数，无效分数用-inf表示。
        - 如果output_mode为'hit':
            - new_indices: 形状为(num_hitted, 3)的矩阵，每行格式为[qry_index, ref_index, top_k_pos]，数据类型为long。
            - new_scores: 形状为(num_hitted,)的张量，包含有效命中的相似度分数，数据类型为float。
    - 所有返回结果均按照scores进行排序，从高到低。
    """
    
    # 自动推断工作设备
    _work_device = resolve_device(work_device, query_queue[0].device)
    # 自动推断输出设备
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)
    
    # 构造工作函数
    if _work_device.type.startswith('cuda'):
        work_func = partial(
            emb_similarity_search_cuda,
            sim_operator=operator,
            top_k=top_k,
            chunk_size=chunk_size,
            output_mode=output_mode,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        work_func = partial(
            emb_similarity_search_cpu,
            sim_operator=operator,  
            top_k=top_k,
            chunk_size=chunk_size,
            output_mode=output_mode,
            work_device=_work_device,
            output_device=_output_device
        )
    
    # 任务分发
    queue_bag = db.from_sequence(zip(query_queue, ref_queue), npartitions=num_workers)
    queue_results = queue_bag.map(lambda x: work_func(x[0],x[1]))
    
    # 计算
    results = queue_results.compute(scheduler='threads', num_workers=num_workers)
    
    return results