import torch
from typing import Tuple
from torch import Tensor
from typing import Literal, Optional

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
    adduct_co_occurrence_threshold: int,
    dim: int,
) -> Tensor:
    if I.size(0) == 0:
        return I
    
    # 直接统计实际存在的参考样本
    unique_refs, ref_counts = torch.unique(I[:,dim], return_counts=True)
    
    # 过滤有效参考样本
    valid_refs = unique_refs[ref_counts >= adduct_co_occurrence_threshold]
    
    # 二次过滤原始索引
    return I[torch.isin(I[:,dim], valid_refs)]

@torch.no_grad()
def mz_search_cpu(
    qry_ions: torch.Tensor,
    ref_mzs: torch.Tensor,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs: Optional[torch.Tensor] = None,
    ref_RTs: Optional[torch.Tensor] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120
) -> torch.Tensor:

    all_indices = []
    # 遍历查询分块
    for q_idx, q_chunk in enumerate(qry_ions.split(chunk_size)):
        q_offset = q_idx * chunk_size

        current_ref_offset = 0  # 参考分块累计偏移
        chunk_indices = []

        # 遍历参考分块
        for r_chunk in ref_mzs.split(chunk_size):
            ref_size = r_chunk.size(0)

            # 计算质量差矩阵 (q_chunk_size, ref_chunk_size)
            Q, R = broadcast(q_chunk, r_chunk)
            delta = torch.abs(Q - R)

            # 处理ppm转换
            if mz_tolerance_type == 'ppm':
                delta = delta * (1e6 / R.clamp(min=1e-6))

            # 生成质量过滤掩码
            mz_mask = delta <= mz_tolerance

            # 处理RT过滤
            if query_RTs is not None and ref_RTs is not None:
                rt_q = query_RTs[q_offset:q_offset+q_chunk.size(0)]
                rt_r = ref_RTs[current_ref_offset:current_ref_offset+ref_size]
                rt_q, rt_r = broadcast(rt_q, rt_r)
                rt_delta = torch.abs(rt_q - rt_r)
                rt_delta = rt_delta.reshape(*rt_delta.shape, *tuple(1 for _ in range(delta.ndim-rt_delta.ndim)))
                mz_mask &= rt_delta <= RT_tolerance

            # 计算局部索引并应用偏移
            local_indices = mz_mask.nonzero(as_tuple=False)
            if local_indices.size(0) > 0:
                # 使用 indices_offset 方法进行偏移
                global_indices = indices_offset(local_indices, q_offset, current_ref_offset)
                chunk_indices.append(global_indices)

            current_ref_offset += ref_size

        # 合并当前查询分块结果
        if chunk_indices:
            all_indices.append(torch.cat(chunk_indices, dim=0))
            
    I = torch.cat(all_indices, dim=0) if all_indices else torch.empty((0,2), dtype=torch.long)
    
    if adduct_co_occurrence_threshold > 1:
        I = adduct_co_occurrence_filter(I, adduct_co_occurrence_threshold, dim=len(qry_ions.shape))
        
    if I.size(0) == 0:
        I = I.reshape(0,len(qry_ions.shape)+len(ref_mzs.shape))

    return I

@torch.no_grad()
def mz_search_gpu(
    qry_ions: torch.Tensor,
    ref_mzs: torch.Tensor,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs: Optional[torch.Tensor] = None,
    ref_RTs: Optional[torch.Tensor] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120
) -> torch.Tensor:

    compute_stream = torch.cuda.current_stream()
    transfer_stream = torch.cuda.Stream()
    all_indices = []

    q_chunks = list(qry_ions.split(chunk_size))
    r_chunks = list(ref_mzs.split(chunk_size))

    for q_idx, q_chunk in enumerate(q_chunks):
        q_offset = q_idx * chunk_size
        current_ref_offset = 0
        chunk_indices = []

        current_r = r_chunks[0].clone()
        for r_idx in range(len(r_chunks)):
            next_r = r_chunks[r_idx+1].clone() if r_idx+1 < len(r_chunks) else None

            with torch.cuda.stream(compute_stream):
                # 计算当前分块
                Q, R = broadcast(q_chunk, current_r)
                delta = torch.abs(Q - R)

                if mz_tolerance_type == 'ppm':
                    delta = delta * (1e6 / R.clamp(min=1e-6))

                mz_mask = delta <= mz_tolerance

                # RT过滤
                if query_RTs is not None and ref_RTs is not None:
                    rt_q = query_RTs[q_offset:q_offset+q_chunk.size(0)]
                    rt_r = ref_RTs[current_ref_offset:current_ref_offset+current_r.size(0)]
                    rt_q, rt_r = broadcast(rt_q, rt_r)
                    rt_delta = torch.abs(rt_q - rt_r)
                    rt_delta = rt_delta.reshape(*rt_delta.shape, *tuple(1 for _ in range(delta.ndim-rt_delta.ndim)))
                    mz_mask &= rt_delta <= RT_tolerance

                # 生成全局索引
                local_indices = mz_mask.nonzero(as_tuple=False)
                if local_indices.size(0) > 0:
                    # 使用 indices_offset 方法进行偏移
                    global_indices = indices_offset(local_indices, q_offset, current_ref_offset)
                    chunk_indices.append(global_indices.cpu())

            # 异步加载下个分块
            if next_r is not None:
                with torch.cuda.stream(transfer_stream):
                    next_r = next_r.to(qry_ions.device, non_blocking=True)

            current_ref_offset += current_r.size(0)
            current_r = next_r
            compute_stream.wait_stream(transfer_stream)

        # 收集当前查询分块结果
        if chunk_indices:
            all_indices.append(torch.cat(chunk_indices, dim=0))

    torch.cuda.synchronize()
    I = torch.cat(all_indices, dim=0) if all_indices else torch.empty((0,2), dtype=torch.long)
    
    if adduct_co_occurrence_threshold > 1:
        I = adduct_co_occurrence_filter(I, adduct_co_occurrence_threshold, dim=len(qry_ions.shape))
        
    if I.size(0) == 0:
        I = I.reshape(0,len(qry_ions.shape)+len(ref_mzs.shape))
    
    return I

def mz_search(
    qry_ions: torch.Tensor,
    ref_mzs: torch.Tensor,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    query_RTs: Optional[torch.Tensor] = None,
    ref_RTs: Optional[torch.Tensor] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
    chunk_size: int = 5120
) -> torch.Tensor:
    
    if qry_ions.device != ref_mzs.device:
        raise ValueError("Input tensors must be on the same device")
    
    if query_RTs is not None and ref_RTs is not None:
        if query_RTs.device != ref_RTs.device:
            raise ValueError("RT tensors must be on the same device")
    
    if qry_ions.device.type == 'cpu':
        return mz_search_cpu(qry_ions, ref_mzs, mz_tolerance, mz_tolerance_type,
                            query_RTs, ref_RTs, RT_tolerance, 
                            adduct_co_occurrence_threshold, chunk_size)
    else:
        return mz_search_gpu(qry_ions, ref_mzs, mz_tolerance, mz_tolerance_type,
                            query_RTs, ref_RTs, RT_tolerance, 
                            adduct_co_occurrence_threshold, chunk_size)