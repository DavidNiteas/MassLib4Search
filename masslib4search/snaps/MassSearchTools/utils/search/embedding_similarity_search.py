import torch
from ..torch_device import resolve_device
from ..similarity.embedding_similarity import (
    cosine,
    EmbbedingSimilarityOperator,
    CosineOperator,
)
from typing import Tuple,Callable,Optional,Union,Literal
    
@torch.no_grad()
def similarity_search_cpu(
    query: torch.Tensor, # shape: (n_q, dim), dtype: float32
    ref: torch.Tensor, # shape: (n_r, dim), dtype: float32
    top_k: Optional[int] = None,
    chunk_size: int = 5120,
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 设备配置
    output_device = output_device or work_device
    
    results = []
    indices_list = []

    # 分块处理查询集（流式传输）
    for q_chunk in query.split(chunk_size):
        # 即时转移设备（自动处理non_blocking）
        q_work = q_chunk.to(work_device)
        
        current_scores = None
        current_indices = None

        # 分块处理参考集
        for r_idx, r_chunk in enumerate(ref.split(chunk_size)):
            r_work = r_chunk.to(work_device)  # 即时转移设备
            
            # 计算相似度
            sim = sim_operator(q_work, r_work)
            
            # 生成索引（在工作设备创建）
            start_idx = r_idx * chunk_size
            indices = torch.arange(
                start_idx,
                start_idx + r_work.size(0),
                device=work_device
            )

            # 结果合并逻辑
            if top_k is not None:
                # TopK模式
                chunk_scores, chunk_indices = torch.topk(sim, min(top_k, sim.size(1)), dim=1)
                chunk_indices = indices[chunk_indices]

                if current_scores is not None:
                    combined_scores = torch.cat([current_scores, chunk_scores], dim=1)
                    combined_indices = torch.cat([current_indices, chunk_indices], dim=1)
                    current_scores, top_pos = torch.topk(combined_scores, top_k, dim=1)
                    current_indices = combined_indices.gather(1, top_pos)
                else:
                    current_scores, current_indices = chunk_scores, chunk_indices
            else:
                # 全量模式
                current_scores = sim if current_scores is None else torch.cat([current_scores, sim], dim=1)
                current_indices = indices.expand_as(sim) if current_indices is None else torch.cat(
                    [current_indices, indices.expand_as(sim)], dim=1)
                
        # 处理完成后转移至output_device
        final_scores = current_scores.to(output_device, non_blocking=True)
        final_indices = current_indices.to(output_device, non_blocking=True)
        
        # 全量排序（仅在需要时执行）
        if top_k is None:
            sorted_scores, sorted_indices = torch.sort(final_scores, dim=1, descending=True)
            results.append(sorted_scores)
            indices_list.append(sorted_indices)
        else:
            results.append(final_scores)
            indices_list.append(final_indices)

    return torch.cat(results, dim=0), torch.cat(indices_list, dim=0)

@torch.no_grad()
def similarity_search_gpu(
    query: torch.Tensor, # shape: (n_q, dim), dtype: float32
    ref: torch.Tensor, # shape: (n_r, dim), dtype: float32
    top_k: Optional[int] = None,
    chunk_size: int = 5120,
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 初始化三流
    input_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    output_stream = torch.cuda.Stream()
    
    # 设备配置
    output_device = output_device or torch.device("cpu")
    is_same_device = work_device == output_device

    # 数据预处理
    with torch.cuda.stream(input_stream):
        query = query.to(work_device)
        ref = ref.to(work_device)

    all_scores, all_indices = [], []
    torch.cuda.synchronize()

    # 主处理循环
    for q_idx, q_chunk in enumerate(query.split(chunk_size)):
        # 初始化当前查询块的存储
        current_scores = None
        current_indices = None
        
        # 预加载第一个参考块
        r_chunks = list(ref.split(chunk_size))
        with torch.cuda.stream(input_stream):
            current_r = r_chunks[0].to(work_device, non_blocking=True)

        # 创建同步事件
        compute_done_events = [torch.cuda.Event() for _ in r_chunks]

        # 处理参考块序列
        for i in range(len(r_chunks)):
            # 预加载下一个参考块
            next_r = r_chunks[i+1] if i+1 < len(r_chunks) else None
            if next_r is not None:
                with torch.cuda.stream(input_stream):
                    next_r = next_r.to(work_device, non_blocking=True)

            # 计算块处理
            with torch.cuda.stream(compute_stream):
                # 等待数据就绪
                compute_stream.wait_stream(input_stream)

                # 执行相似度计算
                sim = sim_operator(q_chunk, current_r)

                # 生成索引
                indices = torch.arange(
                    i * chunk_size,
                    i * chunk_size + current_r.size(0),
                    device=work_device
                )

                # 结果合并逻辑
                if top_k is not None:
                    # 获取当前块的TopK
                    chunk_scores, chunk_indices = torch.topk(
                        sim, 
                        min(top_k, sim.size(1)), 
                        dim=1
                    )
                    chunk_indices = indices[chunk_indices]

                    # 合并历史结果
                    if current_scores is not None:
                        combined_scores = torch.cat([current_scores, chunk_scores], dim=1)
                        combined_indices = torch.cat([current_indices, chunk_indices], dim=1)
                        
                        # 保留全局TopK
                        current_scores, top_indices = torch.topk(combined_scores, top_k, dim=1)
                        current_indices = torch.gather(combined_indices, 1, top_indices)
                    else:
                        current_scores, current_indices = chunk_scores, chunk_indices
                else:
                    # 全量结果累积
                    current_scores = sim if current_scores is None else torch.cat([current_scores, sim], dim=1)
                    current_indices = (
                        indices.expand_as(sim) 
                        if current_indices is None 
                        else torch.cat([current_indices, indices.expand_as(sim)], dim=1)
                    )

                # 记录计算完成事件
                compute_done_events[i].record()

            # 更新参考块
            current_r = next_r

        # 最终结果处理
        with torch.cuda.stream(compute_stream):
            if top_k is None:
                # 全局排序
                sorted_scores, sorted_indices = torch.sort(current_scores, dim=1, descending=True)
                current_scores, current_indices = sorted_scores, sorted_indices

        # 异步传输结果
        with torch.cuda.stream(output_stream):
            # 等待最后一个参考块计算完成
            compute_done_events[-1].synchronize()
            
            if is_same_device:
                all_scores.append(current_scores)
                all_indices.append(current_indices)
            else:
                all_scores.append(current_scores.to(output_device, non_blocking=True))
                all_indices.append(current_indices.to(output_device, non_blocking=True))

    # 全局同步并返回结果
    torch.cuda.synchronize()
    return (
        torch.cat(all_scores, dim=0), 
        torch.cat(all_indices, dim=0)
    )

def similarity_search(
    query: torch.Tensor,
    ref: torch.Tensor,
    top_k: int = None,
    chunk_size: int = 5120,
    sim_operator: EmbbedingSimilarityOperator = CosineOperator,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    operator_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 自动推断工作设备
    _work_device = resolve_device(work_device, query.device)
    # 自动推断输出设备
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)
    
    if query.device.type == 'cpu':
        return similarity_search_cpu(
            query, ref, top_k, chunk_size, operator, _work_device, _output_device
        )
    else:
        return similarity_search_gpu(
            query, ref, top_k, chunk_size, operator, _work_device, _output_device
        )