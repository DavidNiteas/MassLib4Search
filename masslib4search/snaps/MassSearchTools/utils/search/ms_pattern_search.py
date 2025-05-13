import torch
from .ms_peak_search import mz_search,broadcast
from ..spectrum_tools import flatten_sequence,restructure_sequence
import networkx as nx 
from networkx.algorithms.isomorphism import GraphMatcher
import pandas as pd
from numba import njit
import numpy as np
import dask.bag as db
from pydantic import BaseModel,ConfigDict
from typing import List, Dict, Literal, Optional, Union

class SpectrumPatternWrapper(BaseModel):
    
    """
    质谱图模式参考数据容器类

    功能：
        封装预定义的参考质谱图结构及关联的损失值数据，用于质谱图模式匹配算法

    属性：
        graphs (pd.Series[nx.Graph]):
            Series结构存储多个networkx图对象，每个图表示一个参考质谱模式结构
            - 索引：参考图的标识符（通常为字符串或哈希值）
            - 值：nx.Graph对象，包含以下特征：
                * 节点属性可选包含 'mz' (质荷比，float类型)
                * 边属性必须包含 'type' (损失类型，与losses索引关联)

        losses (pd.Series[float]):
            Series结构存储质谱损失值基准数据
            - 索引：损失类型标识符（需与graphs中边的'type'属性对应）
            - 值：浮点数表示的损失值（单位：Da）

    配置：
        model_config = ConfigDict(slots=True, extra='forbid')
        - slots=True: 启用插槽机制，优化内存使用
        - extra='forbid': 禁止额外字段，确保数据模型严格性

    典型使用场景：
        1. 作为 mz_pattern_search 系列函数和PeakPatternSearch工具箱的参考数据输入
        2. 存储预计算的化学物质同位素模式/碎片化模式
    """
    
    model_config = ConfigDict(slots=True,extra='forbid',arbitrary_types_allowed=True)
    
    graphs: pd.Series # pd.Series[nx.Graph]
    losses: pd.Series # PD.Series[float]

@torch.no_grad()
def get_loss_matrix(
    qry_mzs: torch.Tensor, # (n_mzs,)
) -> torch.Tensor: # (n_mzs, n_mzs)
    return torch.sub(*broadcast(qry_mzs, qry_mzs))

def build_qry_graph(
    qry_mzs: torch.Tensor, # (n_mzs,)
    refs: SpectrumPatternWrapper,
    loss_edges: torch.Tensor, # (n_edges, 3[col0: upstream_node_idx, col1: downstream_node_idx, col2: ref_loss_idx])
) -> nx.Graph:
    
    # 创建无向图
    qry_graph = nx.Graph()
    
    # 添加带mz属性的节点
    for idx, mz in enumerate(qry_mzs.cpu().numpy()):
        qry_graph.add_node(idx, mz=float(mz))
    
    # 添加带edge_type属性的边
    if loss_edges.shape[0] > 0:
        li_np = loss_edges.cpu().numpy()
        
        for row in li_np:
            u = int(row[0])
            v = int(row[1])
            ref_loss_idx = int(row[2])
            
            edge_type = refs.losses.index[ref_loss_idx]
            qry_graph.add_edge(u, v, type=edge_type)
    
    return qry_graph
        
def infer_qry_graph(
    qry_mzs: torch.Tensor, # (n_mzs,)
    refs: SpectrumPatternWrapper,
    loss_tolerance: float,
    chunk_size: int = 5120,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Literal['cpu'] = 'cpu',
) -> nx.Graph:
    
    # 获取边信息
    LQ = get_loss_matrix(qry_mzs)
    LR = torch.as_tensor(refs.losses.values, dtype=torch.float32)
    LI = mz_search(LQ,LR,loss_tolerance,"Da",chunk_size=chunk_size, work_device=work_device, output_device=torch.device("cpu"))[0]
    #↑ col0: upstream_node_idx, col1: downstream_node_idx, col2: ref_loss_idx
    
    # 构建查询图
    qry_graph = build_qry_graph(qry_mzs, refs, LI)
    
    return qry_graph

def infer_qry_graph_by_queue(
    qry_mzs_queue: List[torch.Tensor],
    refs_queue: List[SpectrumPatternWrapper],
    loss_tolerance: float,
    chunk_size: int = 5120,
    num_workers: Optional[int] = None,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
) -> List[nx.Graph]:
    qry_mzs_bag = db.from_sequence(qry_mzs_queue)
    LQ_bag = qry_mzs_bag.map(get_loss_matrix)
    LR_bag = db.from_sequence(refs_queue).map(lambda x: torch.as_tensor(x.losses.values, dtype=torch.float32))
    LQR_bag = db.zip(LQ_bag, LR_bag)
    LI_bag = LQR_bag.map(lambda x: mz_search(*x, loss_tolerance, "Da", chunk_size=chunk_size, work_device=work_device, output_device=torch.device("cpu"))[0])
    LI_queue = LI_bag.compute(scheduler='threads', num_workers=num_workers)
    pair_bag = db.from_sequence(zip(qry_mzs_queue, refs_queue, LI_queue))
    graph_bag = pair_bag.map(lambda x: build_qry_graph(*x))
    graph_list = graph_bag.compute(scheduler='processes', num_workers=num_workers)
    return graph_list

@njit(cache=True)
def node_mz_da_match(
    qry_mz: float,
    ref_mz: float,
    mz_tolerance: float,
) -> bool:
    return np.abs(qry_mz - ref_mz) <= mz_tolerance

@njit(cache=True)
def node_mz_ppm_match(
    qry_mz: float,
    ref_mz: float,
    mz_tolerance: float,
) -> bool:
    return (np.abs(qry_mz - ref_mz) / ref_mz) * 1e6 <= mz_tolerance

def mz_graph_search(
    qrys: List[nx.Graph],
    refs: List[SpectrumPatternWrapper],
    mz_tolerance: float,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    num_workers: Optional[int] = None,
) -> List[torch.Tensor]:

    # 动态构建节点匹配策略
    def node_match_builder(qry_node: Dict, ref_node: Dict) -> bool:
        if mz_tolerance <= 0 or "mz" not in ref_node:
            return True
        elif "mz" not in qry_node:
            return False
        if mz_tolerance_type == "ppm":
            return node_mz_ppm_match(qry_node["mz"], ref_node["mz"], mz_tolerance)
        else:
            return node_mz_da_match(qry_node["mz"], ref_node["mz"], mz_tolerance)

    # 边匹配策略（固定匹配类型）
    def edge_match(e1: Dict, e2: Dict) -> bool:
        return e1['type'] == e2['type']

    # 单个匹配任务处理函数
    def match_in_bag(
        qry_graph: nx.Graph, 
        ref_wrapper: SpectrumPatternWrapper,
    ) -> torch.Tensor:
        
        results = []
        
        for i, ref_graph in enumerate(ref_wrapper.graphs):
            
            # 初始化匹配器
            matcher = GraphMatcher(
                qry_graph,
                ref_graph,
                node_match=node_match_builder,
                edge_match=edge_match
            )
            
            # 匹配
            if matcher.subgraph_is_monomorphic():
                results.append(i)
                
        results = torch.as_tensor(results, dtype=torch.int64, device=torch.device("cpu"))
        
        return results

    # 构建并行任务Bag
    qry_bag = db.from_sequence(qrys)
    ref_bag = db.from_sequence(refs)
    bag = db.zip(qry_bag, ref_bag)

    # 执行并行匹配
    results = bag.map(lambda x: match_in_bag(x[0], x[1])).compute(scheduler='processes', num_workers=num_workers)
    
    return results

def mz_pattern_search(
    qry_mzs: List[torch.Tensor],
    refs: SpectrumPatternWrapper,
    loss_tolerance: float,
    mz_tolerance: float,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    chunk_size: int = 5120,
    num_workers: Optional[int] = None,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Literal['cpu'] = 'cpu',
) -> List[torch.Tensor]:
    
    """
    基于质谱图结构的模式匹配搜索（单队列版本）。

    功能：
        对一组查询 m/z 值列表，通过构建质谱图结构并与参考图进行同构匹配，返回匹配的参考图索引。

    参数：
        qry_mzs (List[torch.Tensor]): 
            查询的 m/z 值列表，每个元素为形状 (n_mzs,) 的张量
        refs (SpectrumPatternWrapper): 
            参考图的包装类，包含预定义的质谱图和损失值
        loss_tolerance (float):
            中心丢失边匹配的容差阈值（单位：Da）
        mz_tolerance (float): 
            节点 m/z 匹配容差（单位：ppm 或 Da）
        mz_tolerance_type (Literal['ppm', 'Da']): 
            容差类型，默认为 'ppm'
        chunk_size (int): 
            计算损失矩阵时的分块大小，默认为 5120
        num_workers (Optional[int]): 
            并行计算的工作进程数，None 表示自动选择
        work_device (Union[str, torch.device, Literal['auto']]): 
            张量计算设备，'auto' 表示自动选择 GPU/CPU
        output_device (Literal['cpu']): 
            输出结果强制保留在 CPU

    返回值：
        List[torch.Tensor]: 每个查询对应的匹配参考图索引列表，元素为 int64 类型张量

    流程：
        1. 批量生成查询图（调用 infer_qry_graph_by_queue，使用 loss_tolerance 筛选边）
        2. 并行执行图同构匹配（调用 mz_graph_search）
        3. 返回匹配结果索引
    """
    
    refs_list = [refs] * len(qry_mzs)
    qry_graphs = infer_qry_graph_by_queue(qry_mzs, refs_list, loss_tolerance, chunk_size, num_workers, work_device)
    results = mz_graph_search(qry_graphs, refs_list, mz_tolerance, mz_tolerance_type, num_workers)
    return results

def mz_pattern_search_by_queue(
    qry_mzs_queue: List[List[torch.Tensor]],
    refs_queue: List[SpectrumPatternWrapper],
    loss_tolerance: float,
    mz_tolerance: float,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    chunk_size: int = 5120,
    num_workers: Optional[int] = None,
    work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    output_device: Literal['cpu'] = 'cpu',
) -> List[List[torch.Tensor]]:
    
    """
    基于质谱图结构的模式匹配搜索（多队列批量版本）。

    功能：
        对多个查询队列进行批量处理，通过扁平化输入数据、并行计算和结果重组，实现高效的大规模匹配。

    参数：
        qry_mzs_queue (List[List[torch.Tensor]]): 
            嵌套的查询 m/z 队列，外层列表表示不同批次，内层列表为单批次的查询
        refs_queue (List[SpectrumPatternWrapper]): 
            对应每个批次的参考图包装类列表
        loss_tolerance (float):
            中心丢失边匹配的容差阈值（单位：Da）
        mz_tolerance (float): 
            节点 m/z 匹配容差（单位：ppm 或 Da）
        mz_tolerance_type (Literal['ppm', 'Da']): 
            容差类型，默认为 'ppm'
        chunk_size (int): 
            计算损失矩阵时的分块大小，默认为 5120
        num_workers (Optional[int]): 
            并行计算的工作进程数，None 表示自动选择
        work_device (Union[str, torch.device, Literal['auto']]): 
            张量计算设备，'auto' 表示自动选择 GPU/CPU
        output_device (Literal['cpu']): 
            输出结果强制保留在 CPU

    返回值：
        List[List[torch.Tensor]]: 嵌套的匹配结果，结构与输入 qry_mzs_queue 一致

    流程：
        1. 扁平化输入队列（调用 flatten_sequence）
        2. 批量生成查询图（调用 infer_qry_graph_by_queue，使用 loss_tolerance 筛选边）
        3. 执行并行图匹配（调用 mz_graph_search） 
        4. 按原始结构重组结果（调用 restructure_sequence）
    """
    
    refs_list_queue = [[refs] * len(qry_mzs) for qry_mzs,refs in zip(qry_mzs_queue, refs_queue)]
    flattend_qry_mzs, structure = flatten_sequence(qry_mzs_queue)
    flattend_refs_list, _ = flatten_sequence(refs_list_queue)
    flattend_qry_graphs_queue = infer_qry_graph_by_queue(flattend_qry_mzs, flattend_refs_list, loss_tolerance, chunk_size, num_workers, work_device)
    flattend_results_queue = mz_graph_search(flattend_qry_graphs_queue, flattend_refs_list, mz_tolerance, mz_tolerance_type, num_workers)
    results_queue = restructure_sequence(flattend_results_queue, structure)
    return results_queue