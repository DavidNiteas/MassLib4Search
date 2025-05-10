import torch
import dask
import dask.bag as db
from itertools import accumulate
from typing import List,Tuple,Any

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

def flatten_sequence(nested_seq: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    '''
    将嵌套列表结构压平为单层列表，并记录原始结构
    
    使用Dask并行计算框架实现高性能处理，适用于大规模数据集。通过两步并行操作：
    1. 计算每个子列表的长度作为结构记录
    2. 将嵌套结构完全展开为单层列表

    参数：
    nested_seq -- 嵌套列表结构，例如 [[1,2], [3,4,5], [6]]

    返回：
    (flattened, structure) 二元组：
    - flattened: 压平后的单层列表，例如 [1,2,3,4,5,6]
    - structure: 原始各子列表长度记录，例如 [2,3,1]

    示例：
    >>> flatten_sequence([[1,2], [3,4,5], [6]])
    ([1,2,3,4,5,6], [2,3,1])
    '''
    
    bag = db.from_sequence(nested_seq)
    structure_bag = bag.map(len)
    flattened_bag = bag.flatten()
    flattened,structure = dask.compute(flattened_bag, structure_bag, scheduler='threads')
    return flattened,structure
    
def restructure_sequence(
    flattened: List[Any],
    structure: List[int],
) -> List[List[Any]]:
    """
    根据结构记录将单层列表恢复为原始嵌套结构
    
    本实现使用累积和算法生成精确切片点，时间复杂度为O(n)。包含结构校验机制，
    当结构记录与压平列表长度不匹配时抛出明确异常。

    参数：
    flattened -- 压平后的单层列表
    structure -- 原始结构记录（各子列表长度列表）

    返回：
    恢复后的嵌套列表结构

    异常：
    ValueError -- 当sum(structure) != len(flattened)时抛出

    示例：
    >>> restructure_sequence([1,2,3,4,5], [3,2])
    [[1,2,3], [4,5]]
    
    >>> restructure_sequence([], [0,0])
    [[], []]
    """
    
    if (total := sum(structure)) != len(flattened):
        raise ValueError(f"结构不匹配，总长度应为{total}，实际是{len(flattened)}")
    
    slices = list(accumulate(structure, initial=0))
    
    return [
        flattened[slices[i]:slices[i+1]] 
        for i in range(len(slices)-1)
    ]