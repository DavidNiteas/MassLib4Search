from .ABCs import ToolBox
from pydantic import Field
from ..similarity.operators import (
    EmbbedingSimilarityOperator,
    CosineOperator, DotOperator, JaccardOperator, TanimodoOperator, PearsonOperator,
)
from ..search.embedding_similarity_search import (
    emb_similarity_search_cpu, emb_similarity_search_cuda,
    emb_similarity_search_by_queue,emb_similarity_search
)
import torch
from typing import Tuple, Union, Optional, List, Literal, Type

class EmbeddingSimilaritySearch(ToolBox):
    '''
    嵌入向量相似度搜索工具盒
    
    封装嵌入向量相似度搜索流程，提供配置化的计算参数管理，支持：
    - 设备自动分配策略
    - 批处理并行计算
    - 嵌套数据结构处理
    
    配置参数:
        - `sim_operator`: masslib4search/snaps/MassSearchTools/utils/similarity/operators.py中的相似度算子，以下为预定义的算子
            - `CosineOperator` (默认使用)
            - `DotOperator`
            - `JaccardOperator`
            - `TanimotoOperator`
            - `PearsonOperator`
        - `chunk_size`: 影响内存使用的关键参数（默认5120）
        - `top_k`: 每个查询向量返回前top_k个最相关的参考向量的索引和相似度，默认为None（返回所有相似度）
        - `batch_size`: 批处理大小（默认128）
        - `num_workers`: 数据预处理并行度（默认4线程）
        - `work_device`: 计算设备（自动推断策略：优先使用输入数据所在设备）
        - `output_device`: 默认与计算设备一致
        - `operator_kwargs`: 相似度计算算子的额外参数
    '''

    # 类变量
    func = emb_similarity_search
    func_by_queue = emb_similarity_search_by_queue
    func_cpu = emb_similarity_search_cpu
    func_cuda = emb_similarity_search_cuda
    
    # 实例变量
    sim_operator: Type[EmbbedingSimilarityOperator] = Field(
        CosineOperator,
        title="相似度计算算子",
        description="用于计算嵌入向量之间相似度的算子",
    )
    
    chunk_size: int = Field(
        5120,
        gt=0,
        title="分块大小",
        description="并行计算时每次处理的向量块大小",
    )
    
    top_k: Optional[int] = Field(
        None,
        title="返回的最相似向量数量",
        description="每个查询向量返回前top_k个最相关的参考向量的索引和相似度分数",
    )
    
    batch_size: int = Field(
        128,
        gt=0,
        title="批处理大小",
        description="并行计算批次大小",
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
    
    operator_kwargs: Optional[dict] = Field(
        None,
        title="额外算子参数",
        description="相似度计算算子的额外参数",
    )

    def run(
        self,
        query: torch.Tensor,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        执行嵌入向量相似度搜索（单层结构输入）

        根据类实例的配置参数调用相似度搜索核心逻辑，输入为两个嵌入向量张量

        Args:
            query: 查询向量，形状为 (n_q, dim)，类型为 float32
            ref: 参考向量，形状为 (n_r, dim)，类型为 float32

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 计算得到的相似度矩阵和对应的索引矩阵
                - indices: 参考向量的索引矩阵，形状为 (n_q, top_k) 或 (n_q, n_r)
                - scores: 查询向量与参考向量之间的相似度分数矩阵，形状为 (n_q, top_k) 或 (n_q, n_r)
        '''
        return emb_similarity_search(
            query,
            ref,
            sim_operator=self.sim_operator,
            top_k=self.top_k,
            chunk_size=self.chunk_size,
            work_device=self.work_device,
            output_device=self.output_device,
            operator_kwargs=self.operator_kwargs,
        )
        
    def run_by_queue(
        self,
        query_queue: List[torch.Tensor],
        ref_queue: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''
        执行嵌入向量相似度搜索（嵌套结构输入）

        支持多层级输入结构（如批次数据），自动进行扁平化处理后保持原始数据层级，
        适用于处理多个样本或实验的批量相似度搜索操作

        Args:
            query_queue: 查询向量队列，每个元素形状为 (n_q, dim)，类型为 float32
            ref_queue: 参考向量队列，每个元素形状为 (n_r, dim)，类型为 float32

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: 与输入结构对应的相似度矩阵和索引矩阵列表
                - indices: 参考向量的索引矩阵，形状为 (n_q, top_k) 或 (n_q, n_r)
                - scores: 查询向量与参考向量之间的相似度分数矩阵，形状为 (n_q, top_k) 或 (n_q, n_r)
        '''
        return emb_similarity_search_by_queue(
            query_queue,
            ref_queue,
            sim_operator=self.sim_operator,
            top_k=self.top_k,
            chunk_size=self.chunk_size,
            num_workers=self.num_workers,
            work_device=self.work_device,
            output_device=self.output_device,
            operator_kwargs=self.operator_kwargs,
        )
