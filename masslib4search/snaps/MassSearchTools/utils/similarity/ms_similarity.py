import torch
from ..torch_device import resolve_device
from .operators import SpectramSimilarityOperator,MSEntropyOperator,ms_entropy_similarity
import dask
import dask.bag as db
from itertools import cycle
from typing import List, Callable, Optional, Union, Literal, Dict
    
@torch.no_grad()
def spec_similarity_cpu(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> torch.Tensor:
    
    # 设备推断
    output_device = output_device or work_device

    # 构建Dask Bag
    query_bag = db.from_sequence(query, npartitions=num_dask_workers)
    ref_bag = db.from_sequence(ref, npartitions=num_dask_workers)
    pairs_bag = query_bag.product(ref_bag)
    
    # 计算相似度并转移数据到目标设备
    results_bag = pairs_bag.map(lambda x: sim_operator(x[0].to(work_device), x[1].to(work_device)))
    results_bag = results_bag.map(lambda s: s.to(output_device))

    # 并行计算并合并结果
    results = dask.compute(results_bag, scheduler=dask_mode, num_workers=num_dask_workers)[0]
    similarity_matrix = torch.stack(results).reshape(len(query), len(ref))

    return similarity_matrix
    

@torch.no_grad()
def spec_similarity_cpu_by_queue(
    query: List[List[torch.Tensor]],  # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> List[torch.Tensor]: # Queue[(len(query_block), len(ref_block)]
    
    # 设备推断
    output_device = output_device or work_device

    # 构建配对序列
    bag_queue = []
    for query_block, ref_block in zip(query, ref):
        query_block_bag = db.from_sequence(query_block, npartitions=num_dask_workers)
        ref_block_bag = db.from_sequence(ref_block, npartitions=num_dask_workers)
        pairs_bag = query_block_bag.product(ref_block_bag)
        results_bag = pairs_bag.map(lambda x: sim_operator(x[0].to(work_device), x[1].to(work_device)))
        results_bag = results_bag.map(lambda s: s.to(output_device))
        bag_queue.append(results_bag)
    
    # 使用dask并行计算
    queue_results = dask.compute(bag_queue, scheduler=dask_mode, num_workers=num_dask_workers)[0]
    # 合并结果
    queue_results_bag = db.from_sequence(zip(queue_results,query,ref), npartitions=num_dask_workers)
    queue_results_bag = queue_results_bag.map(lambda x: torch.stack(x[0], dim=0).reshape(len(x[1]), len(x[2])))
    queue_results = queue_results_bag.compute(scheduler=dask_mode, num_workers=num_dask_workers)
    
    return queue_results

@torch.no_grad()
def spec_similarity_cuda(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_cuda_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:

    output_device = output_device or work_device
    torch.cuda.set_device(work_device)
    
    # 为每个worker创建三个专用流
    worker_resources: List[Dict[torch.cuda.Stream,torch.cuda.Event]] = [
        {
            'h2d_stream': torch.cuda.Stream(),  # 数据传入流
            'compute_stream': torch.cuda.Stream(),  # 计算流
            'd2h_stream': torch.cuda.Stream(),  # 结果传出流
            'h2d_event': torch.cuda.Event(),
            'compute_event': torch.cuda.Event(),
        }
        for _ in range(num_cuda_workers)
    ]
    
    # 预分配设备内存
    results = torch.zeros(len(query), len(ref), device=output_device)

    # 异步执行函数
    def _process_pair(q_idx, r_idx, worker_id):
        resources = worker_resources[worker_id]
        
        # 修改后的Stage 1：同时传输query和ref
        with torch.cuda.stream(resources['h2d_stream']):
            q_tensor = query[q_idx].pin_memory().to(work_device, non_blocking=True)
            r_tensor = ref[r_idx].pin_memory().to(work_device, non_blocking=True)
            resources['h2d_event'].record()
        
        # Stage 2保持不变（但使用新传输的r_tensor）
        with torch.cuda.stream(resources['compute_stream']):
            resources['h2d_event'].wait()
            similarity = sim_operator(q_tensor, r_tensor)  # 改用动态传输的ref
            resources['compute_event'].record()
        
        # Stage 3: 结果传回output_device
        with torch.cuda.stream(resources['d2h_stream']):
            resources['compute_event'].wait()
            if output_device != work_device:
                results[q_idx, r_idx] = similarity.to(output_device, non_blocking=True)
            else:
                results[q_idx, r_idx] = similarity

    # 任务调度器
    worker_cycle = cycle(range(num_cuda_workers))
    
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

@torch.no_grad()
def spec_similarity_cuda_by_queue(
    query: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:

    # 如果只有一个队列数据，跳过bag并行，直接调用spec_similarity_cuda
    if len(query) == 1 and len(ref) == 1:
        return [spec_similarity_cuda(
            query[0], ref[0], sim_operator,
            num_cuda_workers=num_cuda_workers,
            work_device=work_device,
            output_device=output_device
        )]
    
    block_bag = db.from_sequence(zip(query, ref), npartitions=num_dask_workers)
    block_bag = block_bag.map(lambda x: spec_similarity_cuda(
        x[0], x[1], sim_operator,
        num_cuda_workers=num_cuda_workers,
        work_device=work_device,
        output_device=output_device
    ))
    results = block_bag.compute(scheduler='threads', num_workers=num_dask_workers)
    
    return results

def spec_similarity(
    query: List[torch.Tensor],
    ref: List[torch.Tensor],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> List[torch.Tensor]:
    
    '''
    执行谱图相似度计算（单层结构输入）

    根据输入参数调用合适的相似度计算核心逻辑，输入为两个谱图向量张量列表。

    Args:
        query: 查询向量列表，每个元素形状为 (n_peaks, 2)，类型为 float32
        ref: 参考向量列表，每个元素形状为 (n_peaks, 2)，类型为 float32
        sim_operator: 用于计算相似度的算子，默认为 MSEntropyOperator
        num_cuda_workers: CUDA并行度，默认为4
        num_dask_workers: Dask并行度，默认为4
        work_device: 计算设备，默认为自动推断
        output_device: 输出设备，默认为自动推断
        dask_mode: Dask的执行模式，默认为None，算子会决定模式
        operator_kwargs: 相似度算子的额外参数，默认为None

    Returns:
        List[torch.Tensor]: 计算得到的相似度矩阵，形状为 (n_q, n_r)，类型为 float32
    '''

    # 设备推断
    _work_device = resolve_device(work_device, query[0].device if query else torch.device('cpu'))
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)

    # 分发实现
    if _work_device.type.startswith('cuda'):
        return spec_similarity_cuda(
            query, ref, operator,
            num_cuda_workers=num_cuda_workers,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return spec_similarity_cpu(
            query, ref, operator,
            num_dask_workers=num_dask_workers,
            work_device=_work_device,
            output_device=_output_device,
            dask_mode=sim_operator.get_dask_mode(dask_mode)
        )
        
def spec_similarity_by_queue(
    query: List[List[torch.Tensor]],
    ref: List[List[torch.Tensor]],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> List[List[torch.Tensor]]:
    
    '''
    执行谱图相似度计算（嵌套结构输入）

    支持多层级输入结构（如批次数据），自动进行扁平化处理后保持原始数据层级，
    适用于处理多个样本或实验的批量相似度计算操作

    Args:
        query: 查询向量队列，每个元素为查询向量列表，每个查询向量形状为 (n_peaks, 2)，类型为 float32
        ref: 参考向量队列，每个元素为参考向量列表，每个参考向量形状为 (n_peaks, 2)，类型为 float32
        sim_operator: 用于计算相似度的算子，默认为 MSEntropyOperator
        num_cuda_workers: CUDA并行度，默认为4
        num_dask_workers: Dask并行度，默认为4
        work_device: 计算设备，默认为自动推断
        output_device: 输出设备，默认为自动推断
        dask_mode: Dask的执行模式，默认为None，算子会决定模式
        operator_kwargs: 相似度算子的额外参数，默认为None

    Returns:
        List[List[torch.Tensor]]: 与输入结构对应的相似度矩阵列表，每个元素形状为 (n_q, n_r)，类型为 float32
    '''
    
    # 设备推断
    _work_device = resolve_device(work_device, query[0][0].device if query else torch.device('cpu'))
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)

    # 分发实现
    if _work_device.type.startswith('cuda'):
        return spec_similarity_cuda_by_queue(
            query, ref, operator,
            num_cuda_workers=num_cuda_workers,
            num_dask_workers=num_dask_workers,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return spec_similarity_cpu_by_queue(
            query, ref, operator,
            num_dask_workers=num_dask_workers,
            work_device=_work_device,
            output_device=_output_device,
            dask_mode=sim_operator.get_dask_mode(dask_mode)
        )