import torch
from .ms_peak_search import mz_search,broadcast
import networkx as nx 
from networkx.algorithms.isomorphism import GraphMatcher
import pandas as pd
from numba import njit
import numpy as np
import dask.bag as db
from typing import List, Dict, Literal, Optional, Hashable, Union

class SpectrumPatternWrapper:
    
    def __init__(
        self,
        graphs: pd.Series, # Series[nx.Graph]
        losses: pd.Series, # Series[float32]
    ) -> None:
        self.graphs = graphs
        self.losses = losses

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
) -> nx.Graph:
    
    # 获取边信息
    LQ = get_loss_matrix(qry_mzs)
    LR = torch.as_tensor(refs.losses.values, dtype=torch.float32, device=work_device)
    LI = mz_search(LQ,LR,loss_tolerance,"Da",chunk_size=chunk_size, work_device=work_device, output_device=torch.device("cpu")) 
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
    LR_bag = db.from_sequence(refs_queue).map(lambda x: torch.as_tensor(x.losses.values, dtype=torch.float32, device=work_device))
    LQR_bag = db.zip(LQ_bag, LR_bag)
    LI_bag = LQR_bag.map(lambda x: mz_search(*x, loss_tolerance, "Da", chunk_size=chunk_size, work_device=work_device, output_device=torch.device("cpu")))
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

def mz_pattern_search(
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