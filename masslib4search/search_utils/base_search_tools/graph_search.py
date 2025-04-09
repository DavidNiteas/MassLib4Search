import torch
from .peak_search import mz_search,broadcast
import networkx as nx 
from networkx.algorithms.isomorphism import GraphMatcher
import pandas as pd
import dask.bag as db
from typing import List, Callable, Dict, Any, Literal

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
    return torch.sub(broadcast(qry_mzs, qry_mzs))
        
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

def graph_match(
    qrys: pd.Series,  # Series[nx.Graph]
    refs: List[AbstructSpectrumGraphWrapper],
    mz_tolerance: float,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
) -> pd.DataFrame:
    """
    并行图匹配核心函数
    
    返回DataFrame结构：
    qry_id | ref_id | matched_nodes | mz_matrix
    """

    # 动态构建节点匹配策略
    def node_match_builder(ref_has_mz: bool) -> Callable:
        if mz_tolerance <= 0 or not ref_has_mz:
            return lambda n1, n2: True
            
        if mz_tolerance_type == "ppm":
            return lambda n1, n2: abs(n1['mz'] - n2['mz']) / n2['mz'] * 1e6 <= mz_tolerance
        else:
            return lambda n1, n2: abs(n1['mz'] - n2['mz']) <= mz_tolerance

    # 边匹配策略（固定匹配类型）
    def edge_match(e1: Dict, e2: Dict) -> bool:
        return e1['type'] == e2['type']

    # 单个匹配任务处理函数
    def match_in_bag(qry_graph: nx.Graph, qry_id: Any, ref_wrapper: AbstructSpectrumGraphWrapper) -> List[Dict]:
        results = []
        
        for ref_id, ref_graph in ref_wrapper.graphs.items():
            # 获取参考图是否包含mz属性
            ref_has_mz = any('mz' in ref_graph.nodes[n] for n in ref_graph.nodes)
            
            # 初始化匹配器
            matcher = GraphMatcher(
                qry_graph,
                ref_graph,
                node_match=node_match_builder(ref_has_mz),
                edge_match=edge_match
            )
            
            # 收集所有匹配
            matches = []
            for mapping in matcher.subgraph_isomorphisms_iter():
                # 提取mz矩阵
                mz_matrix = [
                    [qry_graph.nodes[n]['mz'] for n in sorted_mapping]
                    for sorted_mapping in mapping.values()
                ]
                matches.append({
                    'nodes': list(mapping.keys()),
                    'mz_matrix': mz_matrix
                })
            
            if matches:
                results.append({
                    'qry_id': qry_id,
                    'ref_id': ref_id,
                    'matches': matches
                })
        
        return results

    # 构建并行任务Bag
    bag = db.from_sequence([
        (qry_graph, qry_id, ref)
        for qry_id, qry_graph in qrys.items()
        for ref in refs
    ])

    # 执行并行匹配
    results = bag.map(lambda x: match_in_bag(x[0], x[1], x[2])).compute()

    # 扁平化结果并构建DataFrame
    flat_results = []
    for chunk in results:
        for group in chunk:
            for match in group['matches']:
                flat_results.append({
                    'qry_id': group['qry_id'],
                    'ref_id': group['ref_id'],
                    'matched_nodes': match['nodes'],
                    'mz_matrix': match['mz_matrix']
                })
    
    return pd.DataFrame(flat_results)