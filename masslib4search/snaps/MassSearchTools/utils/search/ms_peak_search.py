import torch
from ..torch_device import resolve_device
from torch import Tensor
import dask.bag as db
from functools import partial
import warnings
from typing import Literal, Optional, Union, Tuple, List

@torch.no_grad()
def broadcast(
    Q: Tensor,
    R: Tensor
) -> Tuple[Tensor, Tensor]:
    
    Q_expanded = Q.view(*Q.shape, *(1,)*R.ndim)
    R_expanded = R.view(*(1,)*Q.ndim, *R.shape)
    
    return Q_expanded, R_expanded

@torch.no_grad()
def get_delta_matrix(
    Q: Tensor,
    R: Tensor
) -> Tensor:
    
    return torch.abs(Q - R)

@torch.no_grad()
def ppm_convert(
    D: Tensor,
    R: Tensor,
) -> Tensor:
    return D * (1e6 / R)

@torch.no_grad()
def get_bool_matrix(
    D: Tensor,
    T: float
) -> Tensor:
    return D <= T

@torch.no_grad()
def get_indices(
    B: Tensor,
) -> Tensor:
    return B.nonzero(as_tuple=False)

@torch.no_grad()
def indices_offset(
    I: Tensor,
    qry_offset: int,
    ref_offset: int,
) -> Tensor:
    I[:, 0] += qry_offset
    I[:, 1] += ref_offset
    return I

@torch.no_grad()
def adduct_co_occurrence_filter(
    I: Tensor,
    threshold: int,
    dim: int,
    D: Tensor,
) -> Tuple[Tensor, Tensor]:
    
    if I.size(0) == 0:
        return (I, D) if D is not None else I
    
    # 统计有效参考样本
    _, ref_counts = torch.unique(I[:, dim], return_counts=True)
    valid_mask = ref_counts[I[:, dim]] >= threshold
    
    # 同步过滤逻辑
    return I[valid_mask], D[valid_mask]

@torch.no_grad()
def mz_search_cpu(
    qry_ions: Tensor,
    ref_mzs: Tensor,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs: Optional[Tensor] = None,
    ref_RTs: Optional[Tensor] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """改进的m/z搜索函数，支持设备管理和分块过滤"""
    output_device = output_device or work_device
    qry_ions = qry_ions.to(work_device)
    ref_mzs = ref_mzs.to(work_device)
    query_RTs = query_RTs.to(work_device) if query_RTs is not None else None
    ref_RTs = ref_RTs.to(work_device) if ref_RTs is not None else None

    all_indices, all_deltas = [], []
    q_dim = len(qry_ions.shape)

    for q_idx, q_chunk in enumerate(qry_ions.split(chunk_size)):
        q_offset = q_idx * chunk_size
        current_ref_offset = 0
        chunk_indices, chunk_deltas = [], []

        for r_chunk in ref_mzs.split(chunk_size):
            Q, R = broadcast(q_chunk.to(work_device), r_chunk.to(work_device))
            delta = torch.abs(Q - R)

            if mz_tolerance_type == 'ppm':
                delta = delta * (1e6 / R.clamp(min=1e-6))

            mz_mask = delta <= mz_tolerance

            if query_RTs is not None and ref_RTs is not None:
                rt_q = query_RTs[q_offset:q_offset+len(q_chunk)]
                rt_r = ref_RTs[current_ref_offset:current_ref_offset+len(r_chunk)]
                rt_q, rt_r = broadcast(rt_q, rt_r)
                rt_delta = torch.abs(rt_q - rt_r)
                mz_mask &= rt_delta <= RT_tolerance

            local_indices = mz_mask.nonzero(as_tuple=False)
            if local_indices.size(0) > 0:
                global_indices = indices_offset(local_indices, q_offset, current_ref_offset)
                chunk_indices.append(global_indices)
                chunk_deltas.append(delta[mz_mask])

            current_ref_offset += len(r_chunk)

        if chunk_indices:
            I_chunk = torch.cat(chunk_indices)
            D_chunk = torch.cat(chunk_deltas)

            if adduct_co_occurrence_threshold > 1:
                I_chunk, D_chunk = adduct_co_occurrence_filter(
                    I_chunk, adduct_co_occurrence_threshold, q_dim, D_chunk
                )

            all_indices.append(I_chunk)
            all_deltas.append(D_chunk)

    I = torch.cat(all_indices) if all_indices else torch.empty((0, 2), dtype=torch.long, device=work_device)
    D = torch.cat(all_deltas) if all_deltas else torch.empty((0,), dtype=ref_mzs.dtype, device=work_device)

    return I.to(output_device), D.to(output_device)

@torch.no_grad()
def mz_search_cuda(
    qry_ions: torch.Tensor,
    ref_mzs: torch.Tensor,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs: Optional[torch.Tensor] = None,
    ref_RTs: Optional[torch.Tensor] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]: # Indices_tensor (hitted_num, qry_dim+ref_dim), Delta_tensor (hitted_num,)
    
    # 设备初始化
    output_device = output_device or work_device
    input_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    output_stream = torch.cuda.Stream()
    
    # 数据预迁移
    with torch.cuda.stream(input_stream):
        qry_ions = qry_ions.to(work_device, non_blocking=True)
        ref_mzs = ref_mzs.to(work_device, non_blocking=True)
        query_RTs = query_RTs.to(work_device, non_blocking=True) if query_RTs is not None else None
        ref_RTs = ref_RTs.to(work_device, non_blocking=True) if ref_RTs is not None else None
    
    # 分块管理
    q_chunks = list(qry_ions.split(chunk_size))
    r_chunks = list(ref_mzs.split(chunk_size))
    
    # 流水线控制
    all_indices, all_deltas = [], []
    
    for q_idx, q_chunk in enumerate(q_chunks):
        q_offset = q_idx * chunk_size
        current_ref_offset = 0
        chunk_indices, chunk_deltas = [], []
        
        # 初始化参考分块流水线
        with torch.cuda.stream(input_stream):
            current_r = r_chunks[0].to(work_device, non_blocking=True)
        
        for r_idx in range(len(r_chunks)):
            # 预取下一分块
            next_r = r_chunks[r_idx+1] if r_idx+1 < len(r_chunks) else None
            if next_r is not None:
                with torch.cuda.stream(input_stream):
                    next_r = next_r.to(work_device, non_blocking=True)
            
            # 计算逻辑
            with torch.cuda.stream(compute_stream):
                torch.cuda.current_stream().wait_stream(input_stream)
                
                Q, R = broadcast(q_chunk, current_r)
                delta = torch.abs(Q - R)
                
                if mz_tolerance_type == 'ppm':
                    delta = delta * (1e6 / R.clamp(min=1e-6))
                
                mz_mask = delta <= mz_tolerance
                
                if query_RTs is not None and ref_RTs is not None:
                    rt_q = query_RTs[q_offset:q_offset+len(q_chunk)]
                    rt_r = ref_RTs[current_ref_offset:current_ref_offset+len(current_r)]
                    rt_q, rt_r = broadcast(rt_q, rt_r)
                    rt_delta = torch.abs(rt_q - rt_r)
                    mz_mask &= rt_delta <= RT_tolerance
                
                local_indices = mz_mask.nonzero(as_tuple=False)
                if local_indices.size(0) > 0:
                    global_indices = indices_offset(local_indices, q_offset, current_ref_offset)
                    chunk_indices.append(global_indices)
                    chunk_deltas.append(delta[mz_mask])
            
            # 结果回传
            with torch.cuda.stream(output_stream):
                torch.cuda.current_stream().wait_stream(compute_stream)
                if chunk_indices:
                    all_indices.append(torch.cat(chunk_indices).to(output_device, non_blocking=True))
                    all_deltas.append(torch.cat(chunk_deltas).to(output_device, non_blocking=True))
            
            current_ref_offset += current_r.size(0)
            current_r = next_r
        
        # 分块内加合物过滤
        with torch.cuda.stream(compute_stream):
            if adduct_co_occurrence_threshold > 1 and chunk_indices:
                I_chunk = torch.cat(chunk_indices)
                D_chunk = torch.cat(chunk_deltas)
                I_chunk, D_chunk = adduct_co_occurrence_filter(
                    I_chunk, adduct_co_occurrence_threshold, 
                    len(qry_ions.shape), D=D_chunk
                )
                chunk_indices = [I_chunk]
                chunk_deltas = [D_chunk]
    
    # 最终同步
    torch.cuda.synchronize()
    
    # 结果整合
    with torch.cuda.stream(output_stream):
        I = torch.cat(all_indices) if all_indices else torch.empty((0,2), dtype=torch.long, device=output_device)
        D = torch.cat(all_deltas) if all_deltas else torch.empty((0,), dtype=ref_mzs.dtype, device=output_device)
        
        # 维度校正
        if I.size(0) == 0:
            I = I.reshape(0, len(qry_ions.shape)+len(ref_mzs.shape))
            D = D.reshape(0)
    
    return I, D

def mz_search(
    qry_ions: torch.Tensor,
    ref_mzs: torch.Tensor,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs: Optional[torch.Tensor] = None,
    ref_RTs: Optional[torch.Tensor] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    执行基于 m/z 和保留时间（RT）的质谱数据匹配搜索，自动选择 CPU 或 CUDA 实现。
    
    参数:
        qry_ions (Tensor): 查询离子（目标）的 m/z 值张量，形状为 (N_q,)
        ref_mzs (Tensor): 参考库（数据库）的 m/z 值张量，形状为 (N_r,)
        mz_tolerance (float): m/z 容差阈值，默认 3（ppm或Da）
        mz_tolerance_type (Literal): 容差类型，'ppm'（百万分比）或 'Da'（绝对质量差）
        query_RTs (Optional[Tensor]): 查询离子的保留时间（RT）张量，形状需与 qry_ions 一致
        ref_RTs (Optional[Tensor]): 参考库的保留时间（RT）张量，形状需与 ref_mzs 一致
        RT_tolerance (float): RT 容差阈值（单位需与输入一致），默认 0.1
        adduct_co_occurrence_threshold (int): 加合物共现过滤阈值，仅保留出现次数≥此值的匹配
        chunk_size (int): 分块处理大小，用于内存优化，默认 5120
        work_device (Union): 计算设备（'auto' 自动推断，或指定 torch.device）
        output_device (Union): 输出设备（'auto' 自动推断，或指定 torch.device）

    返回:
        Tuple[Tensor, Tensor]:
            - 匹配索引张量：形状为 (M, 2)，每行为 (qry_idx, ref_idx)
            - 匹配误差张量：形状为 (M,)，对应每个匹配的 m/z 误差（ppm或Da）

    功能:
        1. 自动根据设备选择 CPU/GPU 实现
        2. 分块广播计算 m/z 和 RT 的绝对误差
        3. 应用容差阈值过滤无效匹配
        4. 执行加合物共现频率过滤（可选）
        5. 结果自动迁移到指定输出设备
    """
    
    # 设备自动推断
    _work_device = resolve_device(work_device, qry_ions.device)
    _output_device = resolve_device(output_device, _work_device)
    
    # 设备分发逻辑
    if _work_device.type.startswith('cuda'):
        return mz_search_cuda(
            qry_ions, ref_mzs, 
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            query_RTs=query_RTs,
            ref_RTs=ref_RTs,
            RT_tolerance=RT_tolerance,
            adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
            chunk_size=chunk_size,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return mz_search_cpu(
            qry_ions, ref_mzs,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            query_RTs=query_RTs,
            ref_RTs=ref_RTs,
            RT_tolerance=RT_tolerance,
            adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
            chunk_size=chunk_size,
            work_device=_work_device,
            output_device=_output_device
        )
        
def mz_search_by_queue(
    qry_ions_queue: List[torch.Tensor],
    ref_mzs_queue: List[torch.Tensor],
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs_queue: Optional[List[Optional[torch.Tensor]]] = None,
    ref_RTs_queue: Optional[List[Optional[torch.Tensor]]] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120,
    num_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
) -> List[Tuple[torch.Tensor,torch.Tensor]]:
    
    """
    批量执行多个 mz_search 任务，支持并行计算。
    
    参数:
        qry_ions_queue (List[Tensor]): 多个查询离子组的列表（每个元素对应一次搜索）
        ref_mzs_queue (List[Tensor]): 多个参考库组的列表（与 qry_ions_queue 一一对应）
        query_RTs_queue (Optional[List): 多个查询 RT 组的列表（可选，需与 qry_ions_queue 对齐）
        ref_RTs_queue (Optional[List): 多个参考 RT 组的列表（可选，需与 ref_mzs_queue 对齐）
        num_workers (int): 并行任务数，默认 4
        其他参数与 mz_search 一致

    返回:
        List[Tuple]: 每个元素为对应任务的 (indices, deltas) 结果元组

    功能:
        1. 使用 Dask 实现多任务并行处理
        2. 自动分发任务到 CPU 线程池或 GPU 流
        3. 支持异构设备任务队列处理
        4. 输入输出队列长度自动校验
        
    """
    
    # 合法性检查
    assert len(qry_ions_queue) == len(ref_mzs_queue)
    if query_RTs_queue is not None:
        assert len(qry_ions_queue) == len(query_RTs_queue)
    if ref_RTs_queue is not None:
        assert len(ref_mzs_queue) == len(ref_RTs_queue)
    
    # 设备自动推断
    _work_device = resolve_device(work_device, qry_ions_queue[0].device)
    _output_device = resolve_device(output_device, _work_device)
    
    if len(qry_ions_queue) > 1:
    
        # 构造工作函数
        if _work_device.type.startswith('cuda'):
            work_func = partial(
                mz_search_cuda,
                mz_tolerance=mz_tolerance,
                mz_tolerance_type=mz_tolerance_type,
                RT_tolerance=RT_tolerance,
                adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
                chunk_size=chunk_size,
                work_device=_work_device,
                output_device=_output_device
            )
        else:
            work_func = partial(
                mz_search_cpu,
                mz_tolerance=mz_tolerance,
                mz_tolerance_type=mz_tolerance_type,
                RT_tolerance=RT_tolerance,
                adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
                chunk_size=chunk_size,
                work_device=_work_device,
                output_device=_output_device
            )
            
        # 任务分发
        query_RTs_queue = query_RTs_queue or [None] * len(qry_ions_queue)
        ref_RTs_queue = ref_RTs_queue or [None] * len(ref_mzs_queue)
        queue_bag = db.from_sequence(zip(qry_ions_queue, ref_mzs_queue, query_RTs_queue, ref_RTs_queue), npartitions=num_workers)
        queue_results = queue_bag.map(lambda x: work_func(x[0],x[1],query_RTs=x[2],ref_RTs=x[3]))
        
        # 计算
        results = queue_results.compute(scheduler='threads', num_workers=num_workers)
        
    elif len(qry_ions_queue) == 1:
        
        results = mz_search(
            qry_ions_queue[0], ref_mzs_queue[0],
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            query_RTs=query_RTs_queue[0] if query_RTs_queue is not None else None,
            ref_RTs=ref_RTs_queue[0] if ref_RTs_queue is not None else None,
            RT_tolerance=RT_tolerance,
            adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
            chunk_size=chunk_size,
            work_device=_work_device,
            output_device=_output_device
        )
        
    else:
        
        warnings.warn("Empty query or reference set, return empty results.")
        results = []
    
    return results