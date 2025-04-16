import torch
from ..torch_device import resolve_device
from .similarity_operator import SpectramSimilarityOperator
import ms_entropy as me
import dask.bag as db
from typing import List, Callable, Optional, Union, Literal

def ms_entropy_similarity(
    query_spec: torch.Tensor, # (n_peaks, 2)
    ref_spec: torch.Tensor, # (n_peaks, 2)
) -> torch.Tensor: # (1,1)
    return torch.tensor([[me.calculate_entropy_similarity(query_spec, ref_spec)]], device=query_spec.device)

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

@torch.no_grad()
def spectrum_similarity_cpu(
    query: List[torch.Tensor],  # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> torch.Tensor:

    # 构建配对序列
    query_bag = db.from_sequence(query, npartitions=num_workers)
    ref_bag = db.from_sequence(ref, npartitions=num_workers)
    pairs_bag = query_bag.product(ref_bag)
    
    # 使用dask并行计算
    pairs_bag = pairs_bag.map(lambda x: sim_operator(x[0].to(work_device), x[1].to(work_device)))
    results = pairs_bag.compute(scheduler=dask_mode, num_workers=num_workers)
    
    # 重组为矩阵
    return torch.stack(results).reshape(len(query), len(ref)).to(output_device or work_device)

@torch.no_grad()
def spectrum_similarity_cuda(
    query: List[torch.Tensor],
    ref: List[torch.Tensor],
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:

    output_device = output_device or work_device
    torch.cuda.set_device(work_device)
    
    # 为每个worker创建三个专用流
    worker_resources = [
        {
            'h2d_stream': torch.cuda.Stream(),  # 数据传入流
            'compute_stream': torch.cuda.Stream(),  # 计算流
            'd2h_stream': torch.cuda.Stream(),  # 结果传出流
            'h2d_event': torch.cuda.Event(),
            'compute_event': torch.cuda.Event(),
        }
        for _ in range(num_workers)
    ]
    
    # 预分配设备内存
    ref_device = [t.to(work_device, non_blocking=True) for t in ref]
    results = torch.zeros(len(query), len(ref), device=output_device)

    # 异步执行函数
    def _process_pair(q_idx, r_idx, worker_id):
        resources = worker_resources[worker_id]
        
        # Stage 1: 数据传输到work_device
        with torch.cuda.stream(resources['h2d_stream']):
            q_tensor = query[q_idx].pin_memory().to(work_device, non_blocking=True)
            resources['h2d_event'].record()
        
        # Stage 2: 计算（等待数据传输完成）
        with torch.cuda.stream(resources['compute_stream']):
            resources['h2d_event'].wait()
            similarity = sim_operator(q_tensor, ref_device[r_idx])
            resources['compute_event'].record()
        
        # Stage 3: 结果传回output_device
        with torch.cuda.stream(resources['d2h_stream']):
            resources['compute_event'].wait()
            if output_device != work_device:
                results[q_idx, r_idx] = similarity.to(output_device, non_blocking=True)
            else:
                results[q_idx, r_idx] = similarity

    # 任务调度器
    from itertools import cycle
    worker_cycle = cycle(range(num_workers))
    
    # 提交任务到流水线
    futures = []
    for q_idx in range(len(query)):
        for r_idx in range(len(ref)):
            worker_id = next(worker_cycle)
            futures.append((q_idx, r_idx, worker_id))
    
    # 启动所有异步任务
    for q_idx, r_idx, worker_id in futures:
        _process_pair(q_idx, r_idx, worker_id)
    
    # 等待所有流完成
    for worker in worker_resources:
        worker['d2h_stream'].synchronize()
    
    return results

def spectrum_similarity(
    query: List[torch.Tensor],
    ref: List[torch.Tensor],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> torch.Tensor:

    # 设备推断
    _work_device = resolve_device(work_device, query[0].device if query else torch.device('cpu'))
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)

    # 分发实现
    if _work_device.type.startswith('cuda'):
        return spectrum_similarity_cuda(
            query, ref, operator,
            num_workers=num_workers,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return spectrum_similarity_cpu(
            query, ref, operator,
            num_workers=num_workers,
            work_device=_work_device,
            output_device=_output_device,
            dask_mode=sim_operator.get_dask_mode(dask_mode)
        )