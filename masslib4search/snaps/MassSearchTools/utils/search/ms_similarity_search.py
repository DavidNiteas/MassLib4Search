import torch
from ..torch_device import resolve_device
from ..similarity.ms_similarity import (
    ms_entropy_similarity,
    SpectramSimilarityOperator,
    MSEntropyOperator,
)
import dask
import dask.bag as db
from typing import Tuple,Callable,Optional,Union,Literal,List

@torch.no_grad()
def spec_similarity_search_cpu(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    top_k: Optional[int] = None,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    output_device = output_device or work_device
    top_k = top_k or len(ref)
    
    # 缓冲区模板
    scores_template = torch.full((top_k,), -float('inf'), 
                                device=work_device, dtype=torch.float32)
    indices_template = torch.full((top_k,), -1, 
                                device=work_device, dtype=torch.long)
    
    # 单query搜索闭包
    def _search_single_query(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 初始化缓冲区
        scores_buf = scores_template.clone()
        indices_buf = indices_template.clone()
        current_count = 0  # 有效结果计数器
        
        q_tensor = q.to(work_device)
        
        for r_idx, r_spec in enumerate(ref):
            score = sim_operator(q_tensor, r_spec.to(work_device))

            # 阶段1：缓冲区未满时的快速写入
            if current_count < top_k:
                scores_buf[current_count] = score
                indices_buf[current_count] = r_idx
                current_count += 1

            # 阶段2：缓冲区已满后的条件替换
            else:
                min_idx = torch.argmin(scores_buf)
                if score > scores_buf[min_idx]:  # 只需比较当前最小值
                    # 定点替换
                    scores_buf[min_idx] = score
                    indices_buf[min_idx] = r_idx
        
        # 后处理缓冲区 （排序）
        valid_part = scores_buf[:current_count]
        sorted_idx = torch.argsort(valid_part, descending=True)
        scores_buf[:current_count] = valid_part[sorted_idx]
        indices_buf[:current_count] = indices_buf[:current_count][sorted_idx]

        return indices_buf.to(output_device),scores_buf.to(output_device)

    # Dask并行处理
    query_bag = db.from_sequence(query, npartitions=num_dask_workers)
    query_bag = query_bag.map(_search_single_query)
    indices_bag = query_bag.pluck(0)
    scores_bag = query_bag.pluck(1)
    indices,scores = dask.compute(indices_bag, scores_bag, scheduler=dask_mode, num_workers=num_dask_workers)
    
    return torch.stack(indices),torch.stack(scores)

@torch.no_grad()
def spec_similarity_search_cpu_by_queue(
    query: List[List[torch.Tensor]],  # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ms_entropy_similarity,
    top_k: Optional[int] = None,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cpu"),
    output_device: Optional[torch.device] = None,
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:

    output_device = output_device or work_device
    top_k = top_k or len(ref)
    
    # 缓冲区模板
    scores_template = torch.full((top_k,), -float('inf'), 
                                device=work_device, dtype=torch.float32)
    indices_template = torch.full((top_k,), -1, 
                                device=work_device, dtype=torch.long)
    
    # 单query搜索闭包
    def _search_single_query(i: int, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 初始化缓冲区
        scores_buf = scores_template.clone()
        indices_buf = indices_template.clone()
        current_count = 0  # 有效结果计数器
        
        q_tensor = q.to(work_device)
        
        for r_idx, r_spec in enumerate(ref[i]):
            score = sim_operator(q_tensor, r_spec.to(work_device))

            # 阶段1：缓冲区未满时的快速写入
            if current_count < top_k:
                scores_buf[current_count] = score
                indices_buf[current_count] = r_idx
                current_count += 1

            # 阶段2：缓冲区已满后的条件替换
            else:
                min_idx = torch.argmin(scores_buf)
                if score > scores_buf[min_idx]:  # 只需比较当前最小值
                    # 定点替换
                    scores_buf[min_idx] = score
                    indices_buf[min_idx] = r_idx
        
        # 后处理缓冲区 （排序）
        valid_part = scores_buf[:current_count]
        sorted_idx = torch.argsort(valid_part, descending=True)
        scores_buf[:current_count] = valid_part[sorted_idx]
        indices_buf[:current_count] = indices_buf[:current_count][sorted_idx]

        return indices_buf.to(output_device), scores_buf.to(output_device)
    
    # 构建配对序列
    bag_queue = []
    for i,query_block in enumerate(query):
        query_block_bag = db.from_sequence(zip([i]*len(query_block), query_block), npartitions=num_dask_workers)
        results_bag = query_block_bag.map(lambda x: _search_single_query(x[0], x[1]))
        indices_bag = results_bag.pluck(0)
        scores_bag = results_bag.pluck(1)
        bag_queue.append((indices_bag, scores_bag))
    
    # 并行搜索
    queue_results = dask.compute(bag_queue, scheduler=dask_mode, num_workers=num_dask_workers)[0]
    
    # 合并结果
    queue_results_bag = db.from_sequence(queue_results, npartitions=num_dask_workers)
    queue_results_bag = queue_results_bag.map(lambda x: (torch.stack(x[0]), torch.stack(x[1])))
    queue_results = queue_results_bag.compute(scheduler="threads", num_workers=num_dask_workers)
    
    return queue_results

@torch.no_grad()
def spec_similarity_search_cuda(
    query: List[torch.Tensor], # List[(n_peaks, 2)]
    ref: List[torch.Tensor], # List[(n_peaks, 2)]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    top_k: Optional[int] = None,
    num_cuda_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    output_device = output_device or work_device
    top_k = top_k or len(ref)
    
    # 初始化CUDA流组（每个worker含3个流）
    stream_groups = [(
        torch.cuda.Stream(device=work_device),  # 数据转移流
        torch.cuda.Stream(device=work_device),  # 计算流
        torch.cuda.Stream(device=work_device)    # 缓冲区流
    ) for _ in range(num_cuda_workers)]

    # 预分配显存资源
    score_buffers = [torch.full((top_k,), -float('inf'), device=work_device) for _ in range(num_cuda_workers)]
    index_buffers = [torch.full((top_k,), -1, device=work_device, dtype=torch.long) for _ in range(num_cuda_workers)]
    event_pool = [torch.cuda.Event() for _ in range(num_cuda_workers*2)]

    # 异步执行容器
    results = [None] * len(query)
    
    for query_idx, q in enumerate(query):
        worker_id = query_idx % num_cuda_workers
        data_stream, compute_stream, buffer_stream = stream_groups[worker_id]
        event_idx = worker_id * 2

        # 阶段1: 异步数据传输
        with torch.cuda.stream(data_stream):
            q_gpu = q.to(work_device, non_blocking=True)
            ref_gpu = [r.to(work_device, non_blocking=True) for r in ref]
            event_pool[event_idx].record(stream=data_stream)

        # 阶段2: 异步计算
        with torch.cuda.stream(compute_stream):
            event_pool[event_idx].wait(stream=compute_stream)  # 等待数据就绪
            scores = []
            for r_idx, r in enumerate(ref_gpu):
                scores.append(sim_operator(q_gpu, r))
            event_pool[event_idx+1].record(stream=compute_stream)

        # 阶段3: 异步缓冲区更新
        with torch.cuda.stream(buffer_stream):
            event_pool[event_idx+1].wait(stream=buffer_stream)  # 等待计算完成
            current_count = 0
            score_buf = score_buffers[worker_id].zero_()
            index_buf = index_buffers[worker_id].zero_()
            
            for r_idx, score in enumerate(scores):
                if current_count < top_k:
                    score_buf[current_count] = score
                    index_buf[current_count] = r_idx
                    current_count += 1
                else:
                    min_idx = torch.argmin(score_buf)
                    if score > score_buf[min_idx]:
                        score_buf[min_idx] = score
                        index_buf[min_idx] = r_idx
            
            # 异步排序
            sorted_idx = torch.argsort(score_buf[:current_count], descending=True)
            score_buf[:current_count] = score_buf[:current_count][sorted_idx]
            index_buf[:current_count] = index_buf[:current_count][sorted_idx]

            # 异步传回结果
            results[query_idx] = (
                index_buf.to(output_device, non_blocking=True),
                score_buf.to(output_device, non_blocking=True),
            )

    # 同步所有流
    torch.cuda.synchronize(work_device)
    
    # 组装最终结果
    return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])

@torch.no_grad()
def spec_similarity_search_cuda_by_queue(
    query: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    ref: List[List[torch.Tensor]], # Queue[List[(n_peaks, 2)]]
    sim_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    top_k: Optional[int] = None,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:

    block_bag = db.from_sequence(zip(query, ref), npartitions=num_dask_workers)
    block_bag = block_bag.map(lambda x: spec_similarity_search_cuda(
        x[0], x[1], sim_operator, top_k, num_cuda_workers, work_device, output_device
    ))
    results = block_bag.compute(scheduler='threads', num_workers=num_dask_workers)
    return results

def spec_similarity_search(
    query: List[torch.Tensor],
    ref: List[torch.Tensor],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    top_k: Optional[int] = None,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    在参考谱图库中执行批量相似度搜索，支持CUDA加速和分布式计算。

    功能说明：
    1. 对每个查询谱图，计算其与所有参考谱图的相似度分数
    2. 自动选择CUDA/CPU计算设备（当work_device='auto'时）
    3. 返回每个查询谱图的top_k最相似结果，包含索引和分数

    参数说明：
    - query      (list[torch.Tensor]) : 查询谱图列表，单谱图形状(n_peaks,2)
    - ref        (list[torch.Tensor]) : 参考谱图列表，单谱图形状(n_peaks,2)
    - sim_operator (callable)         : 相似度计算算子，默认=MSEntropyOperator
    - top_k      (int)               : 返回结果数量，默认=参考库长度
    - num_cuda_workers (int)         : CUDA工作线程数，默认=4
    - num_dask_workers (int)         : Dask工作线程数，默认=4
    - work_device (str|torch.device) : 计算设备，可选['cuda','cpu','auto']，默认='auto'
    - output_device (str|torch.device): 输出设备，默认='auto'
    - dask_mode  (str)               : Dask调度模式，默认=None
    - operator_kwargs (dict)         : 算子额外参数，默认=None

    返回：
    - tuple(
        - indices  (torch.Tensor) : 相似谱图索引，形状(len(query), top_k)
        - scores   (torch.Tensor) : 相似度分数，形状(len(query), top_k) \n
        )

    设备选择：
    - 当work_device='auto'时：优先使用CUDA（若可用），否则使用CPU
    - 输出会自动转换到output_device指定设备
    """

    # 设备推断
    _work_device = resolve_device(work_device, query[0].device if query else torch.device('cpu'))
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)

    # 分发实现
    if _work_device.type.startswith('cuda'):
        return spec_similarity_search_cuda(
            query, ref, operator,
            top_k=top_k,
            num_cuda_workers=num_cuda_workers,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return spec_similarity_search_cpu(
            query, ref, operator,
            top_k=top_k,
            num_dask_workers=num_dask_workers,
            work_device=_work_device,
            output_device=_output_device,
            dask_mode=sim_operator.get_dask_mode(dask_mode)
        )

def spec_similarity_search_by_queue(
    query: List[List[torch.Tensor]],
    ref: List[List[torch.Tensor]],
    sim_operator: SpectramSimilarityOperator = MSEntropyOperator,
    top_k: Optional[int] = None,
    num_cuda_workers: int = 4,
    num_dask_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = None,
    operator_kwargs: Optional[dict] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    
    """
    在参考谱图库中执行批量相似度搜索，支持设备自动选择与并行加速

    功能特性：
    1. 支持CUDA加速计算与Dask分布式计算
    2. 自动设备选择策略（CUDA优先）
    3. 多级并行优化：GPU线程级并行 + Dask任务级并行

    参数说明：
    - query      (list[torch.Tensor]) : 查询谱图列表，单谱图为形状(n_peaks,2)的张量
    - 每行表示一个质谱峰，列0为m/z值，列1为强度值
    - ref        (list[torch.Tensor]) : 参考谱图库，格式同query
    - sim_operator (callable)         : 相似度计算算子，默认=MSEntropyOperator()
    - 应实现__call__(q: Tensor, r: Tensor) -> float 接口
    - top_k      (int)               : 返回的最大结果数，默认=len(ref)
    - num_cuda_workers (int)         : GPU计算流数量，默认=4
    - num_dask_workers (int)         : Dask并行工作节点数，默认=4
    - work_device (str|Device)       : 计算设备['cuda','cpu','auto']，默认='auto'
    - output_device (str|Device)     : 输出设备，默认保持计算结果设备
    - dask_mode  (str)               : Dask调度模式，可选['threads','processes',None]
    - operator_kwargs (dict)         : 算子参数扩展，如{'threshold':0.8}

    返回：
    List[(indices, scores)]:
    - indices  (Tensor) : 相似谱图索引，形状为[查询数, top_k]，按相似度降序排列
    - scores   (Tensor) : 对应相似度分数，形状同indices

    设备策略：
    1. 当work_device='auto'时：
    - 优先使用CUDA（当检测到GPU可用且数据在GPU上）
    - 否则使用CPU并行计算
    2. 输出张量自动转换到output_device指定设备
    """
    
    # 设备推断
    _work_device = resolve_device(work_device, query[0][0].device if query else torch.device('cpu'))
    _output_device = resolve_device(output_device, _work_device)
    
    # 算子生成
    operator = sim_operator.get_operator(_work_device,operator_kwargs)
    
    # 分发实现
    if _work_device.type.startswith('cuda'):
        return spec_similarity_search_cuda_by_queue(
            query, ref, operator,
            top_k=top_k,
            num_cuda_workers=num_cuda_workers,
            num_dask_workers=num_dask_workers,
            work_device=_work_device,
            output_device=_output_device
        )
    else:
        return spec_similarity_search_cpu_by_queue(
            query, ref, operator,
            top_k=top_k,
            num_dask_workers=num_dask_workers,
            work_device=_work_device,
            output_device=_output_device,
            dask_mode=sim_operator.get_dask_mode(dask_mode)
        )