from .ABCs import ToolBox
from pydantic import Field
from ..embedding.spec_binning import (
    binning,binning_by_queue,
    binning_cpu,binning_cuda,
)
import torch
from typing import Tuple,Union,Optional,List,Literal

class SpectrumBinning(ToolBox):
    
    '''
    质谱分箱处理工具盒
    
    封装分箱处理流程，提供配置化的分箱参数管理，支持：
    - 设备自动分配策略
    - 批处理并行计算
    - 嵌套数据结构处理
    
    配置参数:
        binning_window: 分箱范围及精度（默认50-1000 m/z，步长1.0）
        pool_method: 峰值聚合方式（默认求和）
        batch_size: 影响内存使用的关键参数（默认128）
        num_workers: 数据预处理并行度（默认4线程）
        work_device: 计算设备（自动推断策略：优先使用输入数据所在设备）
        output_device: 默认与计算设备一致
        
    方法：
        run(): 单层结构数据处理入口
        run_by_queue(): 嵌套结构数据处理入口
    '''
    
    # class variables
    func = binning
    func_by_queue = binning_by_queue
    func_cpu = binning_cpu
    func_cuda = binning_cuda
    
    # instance variables
    binning_window: Tuple[float, float, float] = Field(
        (50.0, 1000.0, 1.0),
        title="分箱窗口参数",
        description="(最小值, 最大值, 步长)",
        examples=[(0.0, 100.0, 0.5)]
    )
    
    pool_method: Literal['sum', 'max', 'avg'] = Field(
        "sum",
        title="池化方法",
        description="特征聚合方式",
        json_schema_extra={"options": ["sum", "max", "avg"]}
    )
    
    batch_size: int = Field(
        128,
        gt=0,
        title="批处理大小",
        description="并行计算批次大小",
        examples=[64, 128, 256]
    )
    
    num_workers: int = Field(
        4,
        ge=0,
        title="工作线程数",
        description="并行计算的工作线程数"
    )
    
    work_device: Union[str, torch.device, Literal['auto']] = Field(
        'auto',
        title="计算设备",
        description="执行设备自动选择策略",
    )
    
    output_device: Union[str, torch.device, Literal['auto']] = Field(
        'auto',
        title="输出设备",
        description="结果存储设备自动选择策略",
    )
    
    def run(
        self,
        spec_or_mzs: List[torch.Tensor],
        intensities: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        '''
        执行质谱数据分箱处理（单层结构输入）

        根据类实例的配置参数调用分箱核心逻辑，支持两种输入模式：
        - **Spec模式**：输入形状为 `(n_peaks, 2)` 的张量列表（包含 m/z 和强度）
        - **MZ模式**：输入独立的 m/z 列表（形状 `(n_peaks,)`）和强度列表

        Args:
            spec_or_mzs: 质谱数据，可为以下两种形式之一：
                - Spec模式: `[tensor([[mz1, int1], [mz2, int2], ...]), ...]`
                - MZ模式: `[tensor([mz1, mz2, ...]), ...]`（需配合 intensities 参数）
            intensities: 仅用于 MZ 模式的强度值列表，形状需与 spec_or_mzs 一致

        Returns:
            torch.Tensor: 分箱聚合后的强度向量，形状为 `(num_spectrum,num_bins,)`
        '''
        return binning(
            spec_or_mzs,
            intensities,
            self.binning_window,
            self.pool_method,
            self.batch_size,
            self.num_workers,
            self.work_device,
            self.output_device,
        )
        
    def run_by_queue(
        self,
        spec_or_mzs: List[List[torch.Tensor]],
        intensities: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        '''
        执行质谱数据分箱处理（嵌套结构输入）

        支持多层级输入结构（如批次数据），自动进行扁平化处理后保持原始数据层级，
        适用于处理多个样本或实验的批量分箱操作

        Args:
            spec_or_mzs: 嵌套结构的输入数据，支持两种形式：
                - Spec模式: `[[tensor([[mz1,int1],...]), ...], ...]`
                - MZ模式: `[[tensor([mz1, ...]), ...], ...]`（需配合 intensities）
            intensities: 嵌套结构的强度列表，形状需与 spec_or_mzs 完全一致

        Returns:
            List[torch.Tensor]: 与输入结构对应的分箱结果列表，每个元素形状为 `(num_spectrum,num_bins,)`
        '''
        return binning_by_queue(
            spec_or_mzs,
            intensities,
            self.binning_window,
            self.pool_method,
            self.batch_size,
            self.num_workers,
            self.work_device,
            self.output_device,
        )