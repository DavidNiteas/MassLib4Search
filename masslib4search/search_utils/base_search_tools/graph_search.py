import torch
from MassLib4Search.masslib4search.search_utils.base_search_tools.peak_search import mz_search,broadcast
import networkx as nx 
from networkx.algorithms.isomorphism import GraphMatcher
import pandas as pd
from numba import njit
import numpy as np
import dask.bag as db
from typing import List, Dict, Literal, Optional, Hashable

class AbstructSpectrumGraphWrapper:
    
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
        
def infer_qry_graph(
    refs: AbstructSpectrumGraphWrapper,
    qry_mzs: torch.Tensor, # (n_mzs,)
    loss_tolerance: float,
    chunk_size: int = 5120,
) -> nx.Graph:
    
    # 获取边信息
    LQ = get_loss_matrix(qry_mzs)
    LR = torch.as_tensor(refs.losses.values, dtype=torch.float32, device=qry_mzs.device)
    LI = mz_search(LQ,LR,loss_tolerance,"Da",chunk_size=chunk_size) 
    #↑ col0: upstream_node_idx, col1: downstream_node_idx, col2: ref_loss_idx
    
    # 创建无向图
    qry_graph = nx.Graph()
    
    # 添加带mz属性的节点
    for idx, mz in enumerate(qry_mzs.cpu().numpy()):
        qry_graph.add_node(idx, mz=float(mz))
    
    # 添加带edge_type属性的边
    if LI.shape[0] > 0:
        li_np = LI.cpu().numpy()
        
        for row in li_np:
            u = int(row[0])
            v = int(row[1])
            ref_loss_idx = int(row[2])
            
            edge_type = refs.losses.index[ref_loss_idx]
            qry_graph.add_edge(u, v, type=edge_type)
    
    return qry_graph

def infer_qry_graph_by_queue(
    refs_queue: List[AbstructSpectrumGraphWrapper],
    qry_mzs_queue: torch.Tensor, # (n_qrys,n_mzs,)
    loss_tolerance: float,
    chunk_size: int = 5120,
    num_workers: Optional[int] = None,
) -> List[nx.Graph]:
    pair_bag = db.from_sequence(zip(qry_mzs_queue.unbind(0), refs_queue))
    graph_bag = pair_bag.map(lambda x: infer_qry_graph(x[1], x[0], loss_tolerance, chunk_size))
    graph_list = graph_bag.compute(scheduler='threads', num_workers=num_workers)
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

def graph_match(
    qrys: pd.Series,  # Series[nx.Graph]
    refs: List[AbstructSpectrumGraphWrapper],
    mz_tolerance: float,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    num_workers: Optional[int] = None,
) -> pd.DataFrame: # columns: qry_ids, ref_ids

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
        qry_id: Hashable, 
        qry_graph: nx.Graph, 
        ref_wrapper: AbstructSpectrumGraphWrapper,
    ) -> pd.DataFrame: # columns: qry_ids, ref_ids
        
        results = {"qry_ids":[], "ref_ids":[]}
        
        for ref_id, ref_graph in ref_wrapper.graphs.items():
            
            # 初始化匹配器
            matcher = GraphMatcher(
                qry_graph,
                ref_graph,
                node_match=node_match_builder,
                edge_match=edge_match
            )
            
            # 匹配
            if matcher.subgraph_is_monomorphic():
                results['qry_ids'].append(qry_id)
                results['ref_ids'].append(ref_id)
                
        results = pd.DataFrame(results)
        
        return results

    # 构建并行任务Bag
    qry_bag = db.from_sequence(qrys.items())
    ref_bag = db.from_sequence(refs)
    bag = qry_bag.product(ref_bag)

    # 执行并行匹配
    results = bag.map(lambda x: match_in_bag(x[0][0], x[0][1], x[1])).compute(scheduler='processes', num_workers=num_workers)

    # 合并结果
    results = pd.concat(results, ignore_index=True)
    
    return results