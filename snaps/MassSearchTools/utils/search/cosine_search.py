import torch
from typing import Tuple
    
@torch.no_grad()
def cosine_similarity_search_cpu(
    query: torch.Tensor,
    ref: torch.Tensor,
    top_k: int = None,
    chunk_size: int = 5120
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    results = []
    indices_list = []
    
    for q_chunk in query.split(chunk_size):
        current_scores = None
        current_indices = None
        
        for r_idx, r_chunk in enumerate(ref.split(chunk_size)):
            # 计算相似度
            dot_product = q_chunk @ r_chunk.T
            norms = torch.norm(q_chunk, dim=1, keepdim=True) * torch.norm(r_chunk, dim=1)
            sim = dot_product / norms
            
            # 索引偏移
            start_idx = r_idx * chunk_size
            indices = torch.arange(start_idx, start_idx + r_chunk.size(0))
            
            if top_k is not None:
                chunk_scores, chunk_indices = torch.topk(sim, min(top_k, sim.size(1)), dim=1)
                chunk_indices = indices[chunk_indices]
                
                if current_scores is not None:
                    combined = torch.cat([current_scores, chunk_scores], dim=1)
                    combined_indices = torch.cat([current_indices, chunk_indices], dim=1)
                    current_scores, top_indices = torch.topk(combined, top_k, dim=1)
                    current_indices = combined_indices.gather(1, top_indices)
                else:
                    current_scores, current_indices = chunk_scores, chunk_indices
            else:
                # 累积全量结果
                current_scores = sim if current_scores is None else torch.cat([current_scores, sim], dim=1)
                current_indices = indices.expand_as(sim) if current_indices is None else torch.cat([current_indices, indices.expand_as(sim)], dim=1)
        
        # 全量结果排序
        if top_k is None:
            sorted_scores, sorted_indices = torch.sort(current_scores, dim=1, descending=True)
            results.append(sorted_scores)
            indices_list.append(sorted_indices)
        else:
            results.append(current_scores)
            indices_list.append(current_indices)
    
    return torch.cat(results, dim=0), torch.cat(indices_list, dim=0)

@torch.no_grad()
def cosine_similarity_search_gpu(
    query: torch.Tensor,
    ref: torch.Tensor,
    top_k: int = None,
    chunk_size: int = 5120
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    query = query / torch.norm(query, dim=1, keepdim=True)
    ref = ref / torch.norm(ref, dim=1, keepdim=True)

    compute_stream = torch.cuda.current_stream()
    transfer_stream = torch.cuda.Stream()
    
    all_scores = []
    all_indices = []

    for q_chunk in query.split(chunk_size):
        current_scores = None
        current_indices = None
        r_chunks = list(ref.split(chunk_size))
        current_r = r_chunks[0].clone().detach()

        if len(r_chunks) > 1:
            compute_stream.wait_stream(transfer_stream)

        for i in range(len(r_chunks)):
            next_r = r_chunks[i+1].clone().detach() if i+1 < len(r_chunks) else None
            
            with torch.cuda.stream(compute_stream):
                sim = q_chunk @ current_r.T
                start_idx = i * chunk_size
                indices = torch.arange(start_idx, start_idx + current_r.size(0), device=query.device)
                
                if top_k is not None:
                    chunk_scores, chunk_indices = torch.topk(sim, min(top_k, sim.size(1)), dim=1)
                    chunk_indices = indices[chunk_indices]
                    
                    if current_scores is not None:
                        combined_scores = torch.cat([current_scores, chunk_scores], dim=1)
                        combined_indices = torch.cat([current_indices, chunk_indices], dim=1)
                        current_scores, top_indices = torch.topk(combined_scores, top_k, dim=1)
                        current_indices = torch.gather(combined_indices, 1, top_indices)
                    else:
                        current_scores, current_indices = chunk_scores, chunk_indices
                else:
                    current_scores = sim if current_scores is None else torch.cat([current_scores, sim], dim=1)
                    current_indices = indices.expand_as(sim) if current_indices is None else torch.cat([current_indices, indices.expand_as(sim)], dim=1)

            if next_r is not None:
                with torch.cuda.stream(transfer_stream):
                    next_r = next_r.to(query.device, non_blocking=True)

            current_r = next_r
            compute_stream.wait_stream(transfer_stream)
        
        # GPU端排序后传输
        with torch.cuda.stream(compute_stream):
            if top_k is None:
                sorted_scores, sorted_indices = torch.sort(current_scores, dim=1, descending=True)
                all_scores.append(sorted_scores.cpu())
                all_indices.append(sorted_indices.cpu())
            else:
                all_scores.append(current_scores.cpu())
                all_indices.append(current_indices.cpu())
    
    torch.cuda.synchronize()
    return torch.cat(all_scores, dim=0), torch.cat(all_indices, dim=0)

def cosine_similarity_search(
    query: torch.Tensor,
    ref: torch.Tensor,
    top_k: int = None,
    chunk_size: int = 5120,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query.device != ref.device:
        raise ValueError("query and ref must be on the same device")
    
    if query.device.type == 'cpu':
        return cosine_similarity_search_cpu(query, ref, top_k, chunk_size)
    else:
        return cosine_similarity_search_gpu(query, ref, top_k, chunk_size)