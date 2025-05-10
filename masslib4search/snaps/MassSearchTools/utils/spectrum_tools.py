import torch
import dask
import dask.bag as db
from typing import List,Tuple

def split_spectrum_tensor(spec_tensor:torch.Tensor):
    """将二维质谱张量拆分为m/z和强度两个一维张量
    
    参数:
        spec_tensor (Tensor): 形状为(n_peaks, 2)的二维张量
        
    返回:
        (mz, intensities): 元组包含两个一维张量
    """
    mz,intensities = spec_tensor[:, 0],spec_tensor[:, 1] 
    return mz, intensities

def combine_spectrum_tensor(mz:torch.Tensor, intensities:torch.Tensor):
    """将两个一维张量合并为二维质谱张量
    
    参数:
        mz (Tensor): 形状(n_peaks,)的m/z值
        intensities (Tensor): 形状(n_peaks,)的强度值
        
    返回:
        spec_tensor: 形状(n_peaks, 2)的二维张量
    """
    if mz.device != intensities.device:
        intensities = intensities.to(mz.device)
    return torch.stack([mz, intensities], dim=1)

def split_spectrum_by_queue(spec_queue: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """并行拆分质谱张量队列
    
    参数:
        spec_queue (List[torch.Tensor]): 包含多个形状为(n_peaks, 2)质谱张量的列表
    
    返回:
        (mzs_queue, intensities_queue): 包含两个列表的元组，
        - mzs_queue: 多个形状为(n_peaks,)的m/z张量列表
        - intensities_queue: 多个形状为(n_peaks,)的强度张量列表
    """
    bag = db.from_sequence(spec_queue)
    split_bag = bag.map(split_spectrum_tensor)
    mzs_bag = split_bag.pluck(0)
    intensities_bag = split_bag.pluck(1)
    mzs_queue,intensities_queue = dask.compute(mzs_bag, intensities_bag,scheduler='threads')
    return mzs_queue, intensities_queue

def combine_spectrum_by_queue(mzs_queue: List[torch.Tensor], intensities_queue: List[torch.Tensor]) -> List[torch.Tensor]:
    """并行合并m/z和强度队列
    
    参数:
        mz_list (List[torch.Tensor]): 多个形状为(n_peaks,)的m/z张量列表
        intens_list (List[torch.Tensor]): 多个形状为(n_peaks,)的强度张量列表
    
    返回:
        List[torch.Tensor]: 合并后的形状为(n_peaks, 2)质谱张量列表
    """
    paired_bag = db.zip(
        db.from_sequence(mzs_queue),
        db.from_sequence(intensities_queue)
    )
    return paired_bag.map(lambda x: combine_spectrum_tensor(*x)).compute(scheduler='threads')