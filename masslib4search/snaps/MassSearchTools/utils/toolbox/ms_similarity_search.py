from .ABCs import ToolBox
from pydantic import Field
from ..search.ms_similarity_search import (
    spec_similarity_search,
    spec_similarity_search_by_queue,
)
from ..similarity.operators import (
    SpectramSimilarityOperator, MSEntropyOperator,
)
import torch
from typing import Union, Optional, List, Literal, Type, Tuple

class SpectrumSimilaritySearch(ToolBox):
    '''
    谱图相似度搜索工具盒
    
    封装谱图相似度搜索流程，提供配置化的计算参数管理，支持：
    - 设备自动分配策略
    - 批处理并行计算
    - 嵌套数据结构处理
    
    配置参数:
        - sim_operator: masslib4search/snaps/MassSearchTools/utils/similarity/operators.py中的相似度算子，以下为预定义的算子
            - MSEntropyOperator (默认使用)
        - top_k: 返回的最大结果数，默认为参考库长度
        - num_cuda_workers: CUDA并行度（默认4个Worker）
        - num_dask_workers: Dask并行度（默认4线程）
        - work_device: 计算设备（自动推断策略：优先使用输入数据所在设备）
        - output_device: 默认与计算设备一致
        - operator_kwargs: 相似度计算算子的额外参数
        - dask_mode: Dask的执行模式（默认threads）
    '''

    # 类变量
    func = spec_similarity_search
    func_by_queue = spec_similarity_search_by_queue
    
    # 实例变量
    sim_operator: Type[SpectramSimilarityOperator] = Field(
        MSEntropyOperator,
        title="相似度计算算子",
        description="用于计算谱图向量之间相似度的算子",
    )
    
    top_k: Optional[int] = Field(
        None,
        title="返回结果数量",
        description="返回每个查询谱图的top_k最相似结果",
    )
    
    num_cuda_workers: int = Field(
        4,
        ge=0,
        title="CUDA工作Worker数",
        description="使用CUDA时的并行度",
    )
    
    num_dask_workers: int = Field(
        4,
        ge=0,
        title="Dask工作线程数",
        description="使用Dask时的并行度",
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
    
    operator_kwargs: Optional[dict] = Field(
        None,
        title="额外算子参数",
        description="相似度计算算子的额外参数",
    )
    
    dask_mode: Optional[Literal["threads", "processes", "single-threaded"]] = Field(
        "threads",
        title="Dask执行模式",
        description="Dask的执行模式",
    )

    def run(
        self,
        query: List[torch.Tensor],  # List[(n_peaks, 2)]
        ref: List[torch.Tensor],  # List[(n_peaks, 2)]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        执行谱图相似度搜索（单层结构输入）

        根据类实例的配置参数调用相似度搜索核心逻辑，输入为两个谱图向量张量列表

        Args:
            query: 查询向量列表，每个元素形状为 (n_peaks, 2)，类型为 float32
            ref: 参考向量列表，每个元素形状为 (n_peaks, 2)，类型为 float32

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 计算得到的相似谱图索引和分数矩阵，索引和分数形状分别为 (n_q, top_k)，类型为 float32
        '''
        return spec_similarity_search(
            query,
            ref,
            sim_operator=self.sim_operator,
            top_k=self.top_k,
            num_cuda_workers=self.num_cuda_workers,
            num_dask_workers=self.num_dask_workers,
            work_device=self.work_device,
            output_device=self.output_device,
            dask_mode=self.dask_mode,
            operator_kwargs=self.operator_kwargs,
        )
        
    def run_by_queue(
        self,
        query_queue: List[List[torch.Tensor]],  # Queue[List[(n_peaks, 2)]]
        ref_queue: List[List[torch.Tensor]],   # Queue[List[(n_peaks, 2)]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:  # Queue[(len(query_block), len(ref_block))]
        '''
        执行谱图相似度搜索（嵌套结构输入）

        支持多层级输入结构（如批次数据），自动进行扁平化处理后保持原始数据层级，
        适用于处理多个样本或实验的批量相似度搜索操作

        Args:
            query_queue: 查询向量队列，每个元素为查询向量列表，每个查询向量形状为 (n_peaks, 2)，类型为 float32
            ref_queue: 参考向量队列，每个元素为参考向量列表，每个参考向量形状为 (n_peaks, 2)，类型为 float32

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: 与输入结构对应的相似度矩阵列表，每个元素形状为 (n_q, top_k)，类型为 float32
        '''
        return spec_similarity_search_by_queue(
            query_queue,
            ref_queue,
            sim_operator=self.sim_operator,
            top_k=self.top_k,
            num_cuda_workers=self.num_cuda_workers,
            num_dask_workers=self.num_dask_workers,
            work_device=self.work_device,
            output_device=self.output_device,
            dask_mode=self.dask_mode,
            operator_kwargs=self.operator_kwargs,
        )
