import torch
import dask.bag as db
from torch.nested import nested_tensor
from ..torch_device import resolve_device
from typing import Literal,List,Optional,Tuple,Union

@torch.no_grad()
def infer_bin_cells(
    min_mz: float = 50.0,
    max_mz: float = 1000.0,
    bin_size: float = 1,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor: # [num_bins, 2 (start, end)]
    
    starts = torch.arange(start=min_mz, end=max_mz, step=bin_size, device=device)
    
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
    work_device: torch.device = torch.device('cpu'),
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    
    # 参数校验
    assert len(mzs) == len(intensities), "M/Z与强度列表长度不一致"
    if output_device is None:
        output_device = work_device

    # 设备转移检查
    if bin_cells.device != work_device:
        bin_cells = bin_cells.to(work_device)

    def process_batch(batch):
        
        mz_batch, intensity_batch = zip(*batch)
        mz_batch = list(mz_batch)  # 转换为列表
        intensity_batch = list(intensity_batch)  # 转换为列表
        
        nt_mz = nested_tensor(mz_batch,device=work_device)
        nt_intensity = nested_tensor(intensity_batch)
        
        padded_mz = nt_mz.to_padded_tensor(0.0)
        padded_intensity = nt_intensity.to_padded_tensor(0.0)
        
        result = binning_step(padded_mz, padded_intensity, bin_cells, pool_method)
        
        if output_device != work_device:  # 仅当目标设备不同时才转移
            result = result.to(output_device)
        
        return result

    batches = [
        list(zip(mzs[i:i+batch_size], intensities[i:i+batch_size]))
        for i in range(0, len(mzs), batch_size)
    ]

    bag = db.from_sequence(batches, npartitions=num_workers)
    results = bag.map(process_batch).compute(scheduler='threads', num_workers=num_workers)

    return torch.cat(results, dim=0)

@torch.no_grad()
def binning_cuda(
    mzs: List[torch.Tensor],
    intensities: List[torch.Tensor],
    bin_cells: torch.Tensor,
    pool_method: Literal['sum','max', 'avg'] = "sum",
    batch_size: int = 128,
    num_workers: int = 4,
    work_device: torch.device = torch.device("cuda:0"),
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:

    # 参数校验与设备设置
    assert len(mzs) == len(intensities), "M/Z与强度列表长度不一致"
    output_device = output_device or work_device  # 设置默认输出设备

    # 设备转移检查
    if bin_cells.device != work_device:
        bin_cells = bin_cells.to(work_device, non_blocking=True)

    class Worker:
        def __init__(self):
            # 四阶段流定义
            self.transfer_stream = torch.cuda.Stream(device=work_device)  # 数据传输流
            self.padding_stream = torch.cuda.Stream(device=work_device)  # padding流
            self.compute_stream = torch.cuda.Stream(device=work_device)   # 计算流
            self.output_stream = torch.cuda.Stream(device=work_device)    # 结果回传流
            
            # 同步事件
            self.transfer_done = torch.cuda.Event()
            self.padding_done = torch.cuda.Event()
            self.compute_done = torch.cuda.Event()

    workers = [Worker() for _ in range(num_workers)]

    def process_worker(batch, worker: Worker):
        result = None
        
        # 阶段1：数据传输 (CPU->GPU)
        with torch.cuda.stream(worker.transfer_stream):
            # 异步传输原始数据
            mz_batch = [t.to(work_device, non_blocking=True) for t, _ in batch]
            intensity_batch = [t.to(work_device, non_blocking=True) for _, t in batch]
            worker.transfer_done.record()

        # 阶段2：Padding处理
        with torch.cuda.stream(worker.padding_stream):
            worker.transfer_done.wait()
            
            max_len = max(t.shape[0] for t in mz_batch)
            padded_mz = torch.zeros(len(batch), max_len, device=work_device)
            padded_intensity = torch.zeros_like(padded_mz)
            
            for i, (mz, intensity) in enumerate(zip(mz_batch, intensity_batch)):
                padded_mz[i, :len(mz)] = mz
                padded_intensity[i, :len(intensity)] = intensity
            worker.padding_done.record()

        # 阶段3：Binning计算
        with torch.cuda.stream(worker.compute_stream):
            worker.padding_done.wait()
            result = binning_step(padded_mz, padded_intensity, bin_cells, pool_method)
            worker.compute_done.record()

        # 阶段4：结果回传 (GPU->目标设备)
        with torch.cuda.stream(worker.output_stream):
            worker.compute_done.wait()
            if output_device != work_device:  # 仅当目标设备不同时才转移
                result = result.to(output_device, non_blocking=True)
            return result

    # 流水线执行
    results = []
    for batch_idx in range(0, len(mzs), batch_size):
        worker = workers[batch_idx % num_workers]
        batch = list(zip(
            mzs[batch_idx:batch_idx+batch_size],
            intensities[batch_idx:batch_idx+batch_size]
        ))
        results.append(process_worker(batch, worker))

    # 同步所有流并返回结果
    torch.cuda.synchronize()
    return torch.cat(results, dim=0)

def binning(
    mzs: List[torch.Tensor],
    intensities: List[torch.Tensor],
    binning_window: Tuple[float,float,float] = (50.0, 1000.0, 1.0),
    pool_method: Literal['sum','max', 'avg'] = "sum",
    batch_size: int = 128,
    num_workers: int = 4,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Union[str, torch.device, Literal['auto']] = 'auto',
) -> torch.Tensor:
    
    # 参数校验
    assert len(binning_window) == 3, "分箱窗口需要包含三个参数（min_mz, max_mz, bin_size）"
    min_mz, max_mz, bin_size = binning_window
    assert min_mz < max_mz, "最小m/z必须小于最大m/z"
    assert bin_size > 0, "分箱尺寸必须大于0"

    # 自动推断工作设备
    _work_device = resolve_device(work_device, mzs[0].device)
    # 自动推断输出设备
    _output_device = resolve_device(output_device, _work_device)

    # 生成分箱单元格并转移到目标设备
    bin_cells = infer_bin_cells(min_mz, max_mz, bin_size).to(_work_device)

    # 选择执行路径
    if _work_device.type.startswith('cuda'):
        return binning_cuda(
            mzs=mzs,
            intensities=intensities,
            bin_cells=bin_cells,
            pool_method=pool_method,
            batch_size=batch_size,
            num_workers=num_workers,
            work_device=_work_device,
            output_device=_output_device,
        )
    else:
        return binning_cpu(
            mzs=mzs,
            intensities=intensities,
            bin_cells=bin_cells,
            pool_method=pool_method,
            batch_size=batch_size,
            num_workers=num_workers,
            work_device=_work_device,
            output_device=_output_device
        )