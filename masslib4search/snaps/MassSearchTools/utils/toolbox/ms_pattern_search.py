from .ABCs import ToolBox
from pydantic import Field
from ..search.ms_pattern_search import (
    mz_pattern_search, mz_pattern_search_by_queue, SpectrumPatternWrapper
)
from typing import List, Optional, Literal, Union
import torch

class PeakPatternSearch(ToolBox):
    
    """
    基于质谱图结构的模式匹配搜索工具箱
    
    封装质谱图结构的模式匹配搜索流程，提供配置化的计算参数管理，支持：
    - 设备自动分配策略
    - 批处理并行计算
    - 嵌套数据结构处理
    
    请注意，本工具的图匹配基于networkx库，这是一个工作在CPU上的工具，因此即使用户指定了CUDA作为工作设备，也只有构图计算会被放在GPU上进行。
    所以，无论用户制定了何种输出设备，输出结果将强制保留在CPU上。另外，由于networkx是一个Python实现的库，为了破除GIL的影响，本工具采用进程并行，请注意内存占用率。
    
    配置参数:
        - loss_tolerance: 浮点型，中心丢失边匹配的容差阈值（单位：Da）
        - mz_tolerance: 浮点型，m/z容差值（单位：ppm或Da）
        - mz_tolerance_type: 字符串字面量，容差类型，默认'ppm'
        - chunk_size: 整型，分块处理大小，默认5120
        - num_workers: 整型，工作进程数，默认None
        - work_device: 计算设备（'auto' 自动推断，或指定 torch.device）
        - output_device: 输出设备，强制保留在CPU
    """
    
    # 类变量
    func = mz_pattern_search
    func_by_queue = mz_pattern_search_by_queue

    # 实例变量
    loss_tolerance: float = Field(default=None, metadata={'description': '中心丢失边匹配的容差阈值（单位：Da）'})
    mz_tolerance: float = Field(default=3, metadata={'description': '质荷比偏差容忍度'})
    mz_tolerance_type: Literal['ppm', 'Da'] = Field(default='ppm', metadata={'description': '质荷比偏差容忍度类型'})
    chunk_size: int = Field(default=5120, metadata={'description': '并行处理分块大小'})
    num_workers: Optional[int] = Field(default=4, metadata={'description': '工作进程数'})
    work_device: Union[str, torch.device, Literal['auto']] = Field(default='auto', metadata={'description': '工作设备'})
    output_device: Union[str, torch.device, Literal['auto']] = Field(default='cpu', metadata={'description': '输出设备，强制保留在CPU'})

    def run(
        self,
        qry_mzs: List[torch.Tensor],
        refs: SpectrumPatternWrapper,
    ) -> List[torch.Tensor]:
        """
        执行基于质谱图结构的模式匹配搜索（单队列版本）

        Args:
            qry_mzs (List[torch.Tensor]): 查询的 m/z 值列表，每个元素为形状 (n_mzs,) 的张量
            refs (SpectrumPatternWrapper): 参考图的包装类，包含预定义的质谱图和损失值

        Returns:
            List[torch.Tensor]: 每个查询对应的匹配参考图索引列表，元素为 int64 类型张量
        """
        return mz_pattern_search(
            qry_mzs=qry_mzs,
            refs=refs,
            loss_tolerance=self.loss_tolerance,
            mz_tolerance=self.mz_tolerance,
            mz_tolerance_type=self.mz_tolerance_type,
            chunk_size=self.chunk_size,
            num_workers=self.num_workers,
            work_device=self.work_device,
            output_device=self.output_device,
        )

    def run_by_queue(
        self,
        qry_mzs_queue: List[List[torch.Tensor]],
        refs_queue: List['SpectrumPatternWrapper'],
    ) -> List[List[torch.Tensor]]:
        """
        执行基于质谱图结构的模式匹配搜索（多队列批量版本）

        Args:
            qry_mzs_queue (List[List[torch.Tensor]]): 
                嵌套的查询 m/z 队列，外层列表表示不同批次，内层列表为单批次的查询
            refs_queue (List[SpectrumPatternWrapper]): 
                对应每个批次的参考图包装类列表

        Returns:
            List[List[torch.Tensor]]: 嵌套的匹配结果，结构与输入 qry_mzs_queue 一致
        """
        return mz_pattern_search_by_queue(
            qry_mzs_queue=qry_mzs_queue,
            refs_queue=refs_queue,
            loss_tolerance=self.loss_tolerance,
            mz_tolerance=self.mz_tolerance,
            mz_tolerance_type=self.mz_tolerance_type,
            chunk_size=self.chunk_size,
            num_workers=self.num_workers,
            work_device=self.work_device,
            output_device=self.output_device,
        )
