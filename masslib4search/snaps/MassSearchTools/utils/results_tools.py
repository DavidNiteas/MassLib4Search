import torch
from typing import Tuple

def topk_to_hit(indices: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将TopK格式的搜索结果转换为展开的命中列表格式
    Args:
        indices: shape (num_qry, top_k), dtype long，包含参考索引（-1表示无效）
        scores:  shape (num_qry, top_k), dtype float，对应分数（-inf表示无效）
    Returns:
        new_indices: shape (num_hitted, 3)，每行格式 [qry_index, ref_index, top_k_pos]
        new_scores:  shape (num_hitted,)，有效命中的分数
    """
    # 生成查询索引网格
    num_qry, top_k = indices.shape
    qry_idx = torch.arange(num_qry, device=indices.device)[:, None].expand(-1, top_k)
    topk_pos = torch.arange(top_k, device=indices.device)[None, :].expand(num_qry, -1)

    # 创建有效命中掩码（排除ref_index=-1）
    valid_mask = (indices != -1) & (scores != -float('inf'))

    # 提取有效数据
    valid_qry = qry_idx[valid_mask]
    valid_ref = indices[valid_mask]
    valid_topk = topk_pos[valid_mask]
    valid_scores = scores[valid_mask]

    # 组合结果
    new_indices = torch.stack([valid_qry, valid_ref, valid_topk], dim=1)
    return new_indices, valid_scores