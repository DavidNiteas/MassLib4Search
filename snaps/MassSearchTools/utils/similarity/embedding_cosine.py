import torch

@torch.no_grad()
def cosine_similarity_cpu(
    query: torch.Tensor,  # shape: (n_q, dim), dtype: float32
    ref: torch.Tensor,    # shape: (n_r, dim), dtype: float32
    chunk_size: int = 5120
) -> torch.Tensor:        # shape: (n_q, n_r), dtype: float32
    
    # 分块计算逻辑
    results = []
    for q_chunk in query.split(chunk_size):
        chunk_results = []
        
        for r_chunk in ref.split(chunk_size):
            dot_product = q_chunk @ r_chunk.T
            norms = torch.norm(q_chunk, dim=1, keepdim=True) * torch.norm(r_chunk, dim=1)
            chunk_results.append((dot_product / norms))
        
        results.append(torch.cat(chunk_results, dim=1))
    return torch.cat(results)

@torch.no_grad()
def cosine_similarity_gpu(
    query: torch.Tensor,
    ref: torch.Tensor,
    chunk_size: int = 5120
) -> torch.Tensor:
    # 预先归一化
    query = query / torch.norm(query, dim=1, keepdim=True)
    ref = ref / torch.norm(ref, dim=1, keepdim=True)
    
    # 创建双缓冲用的CUDA流
    compute_stream = torch.cuda.current_stream()
    transfer_stream = torch.cuda.Stream()
    
    results = []
    for q_chunk in query.split(chunk_size):
        chunk_results = []
        r_chunks = list(ref.split(chunk_size))
        current_r = r_chunks[0].clone().detach()
        
        # 流同步
        if len(r_chunks) > 1:
            compute_stream.wait_stream(transfer_stream)
        
        for i in range(len(r_chunks)):
            next_r = r_chunks[i+1].clone().detach() if i+1 < len(r_chunks) else None
            
            # 在计算流中执行矩阵乘法
            with torch.cuda.stream(compute_stream):
                sim = q_chunk @ current_r.T
                chunk_results.append(sim.to('cpu', non_blocking=True))
                
            # 在传输流中预取下一批数据
            if next_r is not None:
                with torch.cuda.stream(transfer_stream):
                    next_r = next_r.to(query.device, non_blocking=True)
                    
            current_r = next_r
            compute_stream.wait_stream(transfer_stream)
        
        # 确保所有异步操作完成
        torch.cuda.synchronize()
        results.append(torch.cat(chunk_results, dim=1))
    return torch.cat(results)

def cosine_similarity(
    query: torch.Tensor,
    ref: torch.Tensor,
    chunk_size: int = 5120,
) -> torch.Tensor:
    if query.device != ref.device:
        raise ValueError("query and ref must be on the same device")
    if query.device.type == 'cpu':
        return cosine_similarity_cpu(query, ref, chunk_size)
    else:
        return cosine_similarity_gpu(query, ref, chunk_size)