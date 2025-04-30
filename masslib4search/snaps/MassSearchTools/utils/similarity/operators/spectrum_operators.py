import torch
import ms_entropy as me
from .ABC_operator import SpectramSimilarityOperator

def ms_entropy_similarity(
    query_spec: torch.Tensor, # (n_peaks, 2)
    ref_spec: torch.Tensor, # (n_peaks, 2)
) -> torch.Tensor: # zero-dimensional
    print(query_spec.shape, ref_spec.shape)
    sim = me.calculate_entropy_similarity(query_spec, ref_spec)
    return torch.tensor(sim, device=query_spec.device)

class MSEntropyOperator(SpectramSimilarityOperator):
    
    cpu_kwargs = {
        "ms2_tolerance_in_da":0.02, 
        "ms2_tolerance_in_ppm": -1, 
        "clean_spectra": True,
    }
    dask_mode = "threads" # me.calculate_entropy_similarity是CPU函数，因此默认使用线程池
    
    @classmethod
    def cpu_operator(cls):
        return ms_entropy_similarity
    
    @classmethod
    def cuda_operator(cls):
        raise NotImplementedError(f"{cls.__name__} not supported on CUDA")