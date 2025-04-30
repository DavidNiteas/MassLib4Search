import torch
from .ABC_operator import EmbbedingSimilarityOperator

@torch.no_grad()
def tanimoto(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """Tanimoto系数"""
    vp = torch.sum(va.unsqueeze(-2) * vb.unsqueeze(-3), dim=-1)
    vas = torch.sum(va**2, dim=-1, keepdim=True)
    vbs = torch.sum(vb**2, dim=-1, keepdim=True)
    return vp / (vas + vbs.transpose(-1,-2) - vp + 1e-6)

class TanimodoOperator(EmbbedingSimilarityOperator):
    
    @classmethod
    def cuda_operator(cls):
        return tanimoto
    
    @classmethod
    def cpu_operator(cls):
        return tanimoto

@torch.no_grad()
def cosine(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """余弦相似度"""
    norm_a = torch.norm(va, p=2, dim=-1, keepdim=True)
    norm_b = torch.norm(vb, p=2, dim=-1, keepdim=True)
    return torch.matmul(va, vb.transpose(-1,-2)) / (norm_a * norm_b.transpose(-1,-2) + 1e-6)

class CosineOperator(EmbbedingSimilarityOperator):
    
    @classmethod
    def cuda_operator(cls):
        return cosine
    
    @classmethod
    def cpu_operator(cls):
        return cosine

@torch.no_grad()
def dot(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """点积"""
    return torch.matmul(va, vb.transpose(-1,-2))

class DotOperator(EmbbedingSimilarityOperator):
    
    @classmethod
    def cuda_operator(cls):
        return dot

    @classmethod
    def cpu_operator(cls):
        return dot

@torch.no_grad()
def jaccard(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """Jaccard系数"""
    intersection = torch.logical_and(va.unsqueeze(-2), vb.unsqueeze(-3))
    union = torch.logical_or(va.unsqueeze(-2), vb.unsqueeze(-3))
    return torch.sum(intersection.float(), dim=-1) / (torch.sum(union.float(), dim=-1) + 1e-6)

class JaccardOperator(EmbbedingSimilarityOperator):
    
    @classmethod
    def cuda_operator(cls):
        return jaccard

    @classmethod
    def cpu_operator(cls):
        return jaccard

@torch.no_grad()
def pearson(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """皮尔逊相关系数"""
    mean_a = torch.mean(va, dim=-1, keepdim=True)
    mean_b = torch.mean(vb, dim=-1, keepdim=True)
    centered_a = va - mean_a
    centered_b = vb - mean_b
    numerator = torch.matmul(centered_a, centered_b.transpose(-1,-2))
    denominator = torch.sqrt(
        torch.sum(centered_a**2, dim=-1, keepdim=True) *
        torch.sum(centered_b**2, dim=-1, keepdim=True).transpose(-1,-2)
    )
    return numerator / (denominator + 1e-6)

class PearsonOperator(EmbbedingSimilarityOperator):
    
    @classmethod
    def cuda_operator(cls):
        return pearson

    @classmethod
    def cpu_operator(cls):
        return pearson