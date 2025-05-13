from .ABCs import ToolBox
from pydantic import Field
from ..search.ms_peak_search import (
    mz_search, mz_search_by_queue,
    mz_search_cpu, mz_search_cuda
)
import torch
from typing import Tuple, Union, Optional, List, Literal

class PeakMZSearch(ToolBox):
    
    """
    基于m/z值的质谱峰搜索工具箱
    
    封装m/z峰搜索流程，提供配置化的计算参数管理，支持：
    - 设备自动分配策略
    - 批处理并行计算
    - 嵌套数据结构处理
    
    配置参数:
        - mz_tolerance: 浮点型，m/z容差值，默认3（ppm或Da）
        - mz_tolerance_type: 字符串字面量，容差类型，默认'ppm'
        - RT_tolerance: 浮点型，RT容差值，默认0.1
        - adduct_co_occurrence_threshold: 整型，加合物共现过滤阈值，默认1
        - chunk_size: 整型，分块处理大小，默认5120
        - work_device: 计算设备（'auto' 自动推断，或指定 torch.device）
        - output_device: 输出设备（'auto' 自动推断，或指定 torch.device）
    """
    
    # 类变量
    func = mz_search
    func_by_queue = mz_search_by_queue
    func_cpu = mz_search_cpu
    func_cuda = mz_search_cuda

    # 实例变量
    mz_tolerance: float = Field(
        default=3,
        description="质荷比偏差容忍度（单位：ppm或Da）"
    )
    mz_tolerance_type: Literal['ppm', 'Da'] = Field(
        default='ppm',
        description="质荷比容差类型：ppm（百万分之一）或 Da（道尔顿）"
    )
    RT_tolerance: float = Field(
        default=0.1,
        description="保留时间容差（单位：分钟）"
    )
    adduct_co_occurrence_threshold: int = Field(
        default=1,
        description="加合物共现最小次数阈值（>=此值才视为有效加合物）",
        ge=1
    )
    chunk_size: int = Field(
        default=5120,
        description="数据分块大小（平衡内存使用与计算效率）",
        ge=128
    )
    work_device: Union[str, torch.device, Literal['auto']] = Field(
        default='auto',
        description="计算设备，自动模式（auto）使用输入数据所在的设备"
    )
    output_device: Union[str, torch.device, Literal['auto']] = Field(
        default='auto',
        description="输出设备，auto模式保持与work_device一致"
    )

    def run(
        self,
        qry_ions: torch.Tensor,
        ref_mzs: torch.Tensor,
        qry_RTs: Optional[torch.Tensor] = None,
        ref_RTs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行基于m/z值的质谱峰搜索（单层结构输入）

        Args:
            qry_ions (torch.Tensor): 查询离子（目标）的 m/z 值张量，形状为 (N_q,)
            ref_mzs (torch.Tensor): 参考库（数据库）的 m/z 值张量，形状为 (N_r,)
            qry_RTs (Optional[torch.Tensor], optional): 查询离子的 RT 值张量，形状为 (N_q,)，默认 None
            ref_RTs (Optional[torch.Tensor], optional): 参考库的 RT 值张量，形状为 (N_r,)，默认 None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 匹配索引张量：形状为 (M, 2)，每行为 (qry_idx, ref_idx)
                - 匹配误差张量：形状为 (M,)，对应每个匹配的 m/z 误差（ppm或Da）
        """
        return mz_search(
            qry_ions=qry_ions,
            ref_mzs=ref_mzs,
            mz_tolerance=self.mz_tolerance,
            mz_tolerance_type=self.mz_tolerance_type,
            query_RTs=qry_RTs,
            ref_RTs=ref_RTs,
            RT_tolerance=self.RT_tolerance,
            adduct_co_occurrence_threshold=self.adduct_co_occurrence_threshold,
            chunk_size=self.chunk_size,
            work_device=self.work_device,
            output_device=self.output_device,
        )

    def run_by_queue(
        self,
        qry_ions_queue: List[torch.Tensor],
        ref_mzs_queue: List[torch.Tensor],
        qry_RTs_queue: Optional[List[torch.Tensor]] = None,
        ref_RTs_queue: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行基于m/z值的质谱峰搜索（嵌套结构输入）

        Args:
            qry_ions_queue (List[torch.Tensor]): 多个查询离子组的列表（每个元素对应一次搜索）
            ref_mzs_queue (List[torch.Tensor]): 多个参考库组的列表（与 qry_ions_queue 一一对应）
            qry_RTs_queue (Optional[List[torch.Tensor]], optional): 多个查询离子 RT 值组的列表（与 qry_ions_queue 一一对应），默认 None
            ref_RTs_queue (Optional[List[torch.Tensor]], optional): 多个参考库 RT 值组的列表（与 qry_ions_queue 一一对应），默认 None

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: 每个元素为对应任务的 (indices, deltas) 结果元组
        """

        return mz_search_by_queue(
            qry_ions_queue=qry_ions_queue,
            ref_mzs_queue=ref_mzs_queue,
            mz_tolerance=self.mz_tolerance,
            mz_tolerance_type=self.mz_tolerance_type,
            query_RTs_queue=qry_RTs_queue,
            ref_RTs_queue=ref_RTs_queue,
            RT_tolerance=self.RT_tolerance,
            adduct_co_occurrence_threshold=self.adduct_co_occurrence_threshold,
            chunk_size=self.chunk_size,
            work_device=self.work_device,
            output_device=self.output_device,
        )