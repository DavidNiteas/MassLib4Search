import torch
import dask.bag as db
from torch.nested import nested_tensor
from typing import Literal,List,Optional,Tuple

@torch.no_grad()
def infer_bin_cells(
    min_mz: float = 50.0,
    max_mz: float = 1000.0,
    bin_size: float = 1,
) -> torch.Tensor: # [num_bins, 2 (start, end)]
    
    starts = torch.arange(start=min_mz, end=max_mz, step=bin_size)
    
    ends = starts + bin_size
    
    return torch.stack((starts, ends), dim=1)

@torch.no_grad()
def binning_step(
    padded_mzs: torch.Tensor,
    padded_intensity: torch.Tensor,
    bin_cells: torch.Tensor,
    pool_method: Literal['sum','max', 'avg'] = "sum",
) -> torch.Tensor:
    
    # 生成掩码张量 [n_spec, num_peaks, num_bins]
    mask = (padded_mzs.unsqueeze(-1) >= bin_cells[:, 0]) & (padded_mzs.unsqueeze(-1) < bin_cells[:, 1])
    mask = mask.float()
    
    # 批量池化计算
    if pool_method == "sum":
        return torch.einsum('spb,sp->sb', mask, padded_intensity)
    elif pool_method == "max":
        expanded_intensity = padded_intensity.unsqueeze(-1) * mask  # [n_spec, num_peaks, num_bins]
        return torch.where(mask.any(dim=1), 
                          expanded_intensity.max(dim=1).values, 
                          torch.zeros_like(expanded_intensity[:,0,:]))
    elif pool_method == "avg":
        sum_result = torch.einsum('spb,sp->sb', mask, padded_intensity)
        counts = mask.sum(dim=1)  # [n_spec, num_bins]
        return sum_result / counts.clamp(min=1e-8)
    else:
        raise ValueError(f"Unsupported pooling method: {pool_method}")
    
@torch.no_grad()
def binning_cpu(
    mzs: List[torch.Tensor],
    intensities: List[torch.Tensor],
    bin_cells: torch.Tensor,
    pool_method: Literal['sum','max', 'avg'] = "sum",
    batch_size: int = 128,
    num_workers: int = 4,
) -> torch.Tensor:
    
    # 参数校验
    assert len(mzs) == len(intensities), "M/Z与强度列表长度不一致"

    def process_batch(batch):
        
        mz_batch, intensity_batch = zip(*batch)
        mz_batch = list(mz_batch)  # 转换为列表
        intensity_batch = list(intensity_batch)  # 转换为列表
        
        nt_mz = nested_tensor(mz_batch)
        nt_intensity = nested_tensor(intensity_batch)
        
        padded_mz = nt_mz.to_padded_tensor(0.0)
        padded_intensity = nt_intensity.to_padded_tensor(0.0)
        
        return binning_step(padded_mz, padded_intensity, bin_cells, pool_method)

    batches = [
        list(zip(mzs[i:i+batch_size], intensities[i:i+batch_size]))
        for i in range(0, len(mzs), batch_size)
    ]

    bag = db.from_sequence(batches, npartitions=num_workers)
    results = bag.map(process_batch).compute()

    return torch.cat(results, dim=0)

@torch.no_grad()
def binning_gpu(
    mzs: List[torch.Tensor],
    intensities: List[torch.Tensor],
    bin_cells: torch.Tensor,
    pool_method: Literal['sum','max', 'avg'] = "sum",
    batch_size: int = 128,
    num_workers: int = 4,
) -> torch.Tensor:

    # 参数校验
    assert len(mzs) == len(intensities), "M/Z与强度列表长度不一致"
    device = bin_cells.device
    batch_size = batch_size or len(mzs)
    num_workers = num_workers or 4

    class Worker:
        def __init__(self, device):
            self.copy_stream = torch.cuda.Stream(device=device)  # 数据拷贝流
            self.compute_stream = torch.cuda.Stream(device=device)  # 计算流
            self.event = torch.cuda.Event()  # 用于流间同步

    # 初始化worker池
    workers = [Worker(device) for _ in range(num_workers)]
    
    # 预分配显存 (使用固定内存加速传输)
    bin_cells_gpu = bin_cells.pin_memory().to(device, non_blocking=True)
    pending_batches = []

    def process_worker(batch, worker: Worker):
        # 阶段1：在拷贝流执行数据传输
        with torch.cuda.stream(worker.copy_stream):
            # 异步拷贝数据到GPU
            mz_batch = [t.to(device, non_blocking=True) for t, _ in batch]
            intensity_batch = [t.to(device, non_blocking=True) for _, t in batch]
            
            # 执行padding
            max_len = max(t.shape[0] for t in mz_batch)
            padded_mz = torch.zeros(len(batch), max_len, device=device)
            padded_intensity = torch.zeros_like(padded_mz)

            for i, (mz, intensity) in enumerate(zip(mz_batch, intensity_batch)):
                padded_mz[i, :len(mz)] = mz
                padded_intensity[i, :len(intensity)] = intensity
            
            # 记录事件同步点
            worker.event.record(worker.copy_stream)

        # 阶段2：在计算流执行计算
        with torch.cuda.stream(worker.compute_stream):
            # 等待拷贝流完成
            worker.event.wait(worker.compute_stream)

            return binning_step(padded_mz, padded_intensity, bin_cells_gpu, pool_method)

    # 流水线执行
    for i in range(0, len(mzs), batch_size):
        current_worker = workers[i % num_workers]
        batch = list(zip(mzs[i:i+batch_size], intensities[i:i+batch_size]))
        
        # 提交任务到worker
        future = process_worker(batch, current_worker)
        pending_batches.append(future)

    # 同步所有worker
    for worker in workers:
        torch.cuda.synchronize(worker.copy_stream)
        torch.cuda.synchronize(worker.compute_stream)

    return torch.cat(pending_batches, dim=0)

def binning(
    mzs: List[torch.Tensor],
    intensities: List[torch.Tensor],
    binning_window: Tuple[float,float,float] = (50.0, 1000.0, 1.0),
    pool_method: Literal['sum','max', 'avg'] = "sum",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    device: Literal['cpu', 'cuda'] = "cpu",
) -> torch.Tensor:
    """
    统一分箱处理入口函数
    
    参数：
    - mzs: 质荷比列表，每个元素为形状[num_peaks]的张量
    - intensities: 强度值列表，每个元素为形状[num_peaks]的张量
    - binning_window: (min_mz, max_mz, bin_size) 分箱参数
    - pool_method: 池化方法（sum/max/avg）
    - batch_size: 批处理大小
    - num_workers: 并行工作进程数
    - device: 运行设备（cpu/cuda）
    
    返回：
    - 分箱结果张量，形状为[num_spectra, num_bins]
    """
    
    # 参数校验
    assert len(binning_window) == 3, "分箱窗口需要包含三个参数（min_mz, max_mz, bin_size）"
    min_mz, max_mz, bin_size = binning_window
    assert min_mz < max_mz, "最小m/z必须小于最大m/z"
    assert bin_size > 0, "分箱尺寸必须大于0"
    
    # 生成分箱单元格
    bin_cells = infer_bin_cells(min_mz, max_mz, bin_size)
    
    # 设备相关预处理
    if device != 'cpu':
        bin_cells = bin_cells.to(device)
    
    # 选择执行路径
    if device == 'cpu':
        return binning_cpu(
            mzs=mzs,
            intensities=intensities,
            bin_cells=bin_cells,
            pool_method=pool_method,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        return binning_gpu(
            mzs=mzs,
            intensities=intensities,
            bin_cells=bin_cells,
            pool_method=pool_method,
            batch_size=batch_size,
            num_workers=num_workers
        )