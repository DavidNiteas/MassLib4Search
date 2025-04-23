import torch
from ..torch_device import resolve_device
from ..similarity.ms_similarity import (
    ms_entropy_similarity,
    SpectramSimilarityOperator,
    MSEntropyOperator,
)
from typing import Tuple,Callable,Optional,Union,Literal,List

@torch.no_grad()
def spectrum_similarity_search_cpu(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    top_k: Optional[int] = None,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

@torch.no_grad()
def spectrum_similarity_search_cpu_by_queue(
    query: List[List[torch.Tensor]],  # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pass

@torch.no_grad()
def spectrum_similarity_search_cuda(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    top_k: Optional[int] = None,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

@torch.no_grad()
def spectrum_similarity_search_cuda_by_queue(
    query: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pass

def spectrum_similarity_search(
    query: List[torch.Tensor],
    ref: List[torch.Tensor],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

def spectrum_similarityy_search_by_queue(
    query: List[List[torch.Tensor]],
    ref: List[List[torch.Tensor]],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pass