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
        - `output_mode`: 返回结果的格式，可选'top_k'或'hit'。默认为'top_k'。
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
    
    output_mode: Literal['top_k','hit'] = Field(
        'top_k',
        title="结果模式",
        description="结果返回模式,'top_k'返回top_k个最相似向量的索引和相似度的二维矩阵，'hit'返回所有向量的索引的命中对及相似性",
    )

    def run(
        self,
        query: torch.Tensor,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        执行嵌入向量相似度搜索（单层结构输入）

        根据类实例的配置参数调用相似度搜索核心逻辑，输入为两个嵌入向量张量。
        如果查询集或参考集为空，则返回适当的空张量。

        参数:
            - query: 查询向量集，形状为(n_q, dim)，类型为float32。
            - ref: 参考向量集，形状为(n_r, dim)，类型为float32。

        返回:
            - Tuple[torch.Tensor, torch.Tensor]: 计算得到的相似度矩阵和对应的索引矩阵。
                - 如果output_mode为'top_k':
                    - indices: 形状为(n_q, top_k)的参考向量索引矩阵，数据类型为long。包含每个查询向量最相关的top_k个参考向量的索引，无效索引用-1表示。
                    - scores: 形状为(n_q, top_k)的相似度分数矩阵，数据类型为float。包含每个查询向量与最相关的top_k个参考向量的相似度分数，无效分数用-inf表示。
                - 如果output_mode为'hit':
                    - new_indices: 形状为(num_hitted, 3)的矩阵，每行格式为[qry_index, ref_index, top_k_pos]，数据类型为long。
                    - new_scores: 形状为(num_hitted,)的张量，包含有效命中的相似度分数，数据类型为float。
        '''
        return emb_similarity_search(
            query,
            ref,
            sim_operator=self.sim_operator,
            top_k=self.top_k,
            chunk_size=self.chunk_size,
            output_mode=self.output_mode,
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

        支持多个查询向量集和参考向量集的批量相似度搜索操作。适用于处理多个样本或实验的批量相似度搜索操作。
        如果查询集或参考集为空，则返回适当的空张量。

        参数:
            - query_queue: 查询向量集的列表，每个查询向量集的形状为(n_q, dim)。
            - ref_queue: 参考向量集的列表，每个参考向量集的形状为(n_r, dim)。

        返回:
            - List[Tuple[torch.Tensor, torch.Tensor]]: 一个包含多个元组的列表，每个元组对应于query_queue和ref_queue中的一个查询向量集和参考向量集对。
                - 如果output_mode为'top_k':
                    - indices: 形状为(n_q, top_k)的参考向量索引矩阵，数据类型为long。包含每个查询向量最相关的top_k个参考向量的索引，无效索引用-1表示。
                    - scores: 形状为(n_q, top_k)的相似度分数矩阵，数据类型为float。包含每个查询向量与最相关的top_k个参考向量的相似度分数，无效分数用-inf表示。
                - 如果output_mode为'hit':
                    - new_indices: 形状为(num_hitted, 3)的矩阵，每行格式为[qry_index, ref_index, top_k_pos]，数据类型为long。
                    - new_scores: 形状为(num_hitted,)的张量，包含有效命中的相似度分数，数据类型为float。
        '''
        return emb_similarity_search_by_queue(
            query_queue,
            ref_queue,
            sim_operator=self.sim_operator,
            top_k=self.top_k,
            chunk_size=self.chunk_size,
            output_mode=self.output_mode,
            num_workers=self.num_workers,
            work_device=self.work_device,
            output_device=self.output_device,
            operator_kwargs=self.operator_kwargs,
        )
