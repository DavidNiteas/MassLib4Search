import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import PeakPatternSearch,SpectrumPatternWrapper
import torch
import pandas as pd
import networkx as nx

@pytest.mark.MassSearchTools_utils_search
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'auto', 'output_device': 'cpu'},
    {'work_device': 'cpu', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'cpu'}
])
def test_ms_pattern_search(device_config):
    
    # 构建测试数据
    query_mzs_queue = [[torch.tensor([149.05970595, 277.15869171, 295.1692564, 313.17982108, 373.20095045, 433.22207982, 581.27450931])]] * 2
    graph_0 = nx.Graph()
    graph_0.add_edges_from([
        (0, 1, {'type':0}),
    ])
    graph_1 = nx.Graph()
    graph_1.add_edges_from([
        (0, 1, {'type':0}),
        (1, 2, {'type':0}),
    ])
    graph_2 = nx.Graph()
    graph_2.add_edges_from([
        (0, 1, {'type':0}),
        (1, 2, {'type':0}),
        (2, 3, {'type':0}),
    ])
    losses = pd.Series([60.0],index=[0])
    refs_queue = [SpectrumPatternWrapper(
        graphs=pd.Series([graph_0, graph_1, graph_2]),
        losses=losses
    )] * 2
    
    # 公共参数
    base_params = {
        'loss_tolerance': 0.1,
        'mz_tolerance': 3,
        'mz_tolerance_type': 'ppm',
        'chunk_size': 5120,
        'num_workers': 2,
    }
    
    # 初始化工具箱
    peak_pattern_search_toolbox = PeakPatternSearch(**{**base_params, **device_config})
    
    # 搜索测试
    results = peak_pattern_search_toolbox.run_by_queue(query_mzs_queue, refs_queue)
    
    # 结果验证
    assert len(results) == 2
    assert torch.allclose(results[0][0], results[1][0])
    assert torch.allclose(results[0][0], torch.tensor([0,1]))
    
if __name__ == '__main__':
    
    # 构建测试数据
    query_mzs_queue = [[torch.tensor([149.05970595, 277.15869171, 295.1692564, 313.17982108, 373.20095045, 433.22207982, 581.27450931])]] * 2
    graph_0 = nx.Graph()
    graph_0.add_edges_from([
        (0, 1, {'type':0}),
    ])
    graph_1 = nx.Graph()
    graph_1.add_edges_from([
        (0, 1, {'type':0}),
        (1, 2, {'type':0}),
    ])
    graph_2 = nx.Graph()
    graph_2.add_edges_from([
        (0, 1, {'type':0}),
        (1, 2, {'type':0}),
        (2, 3, {'type':0}),
    ])
    losses = pd.Series([60.0],index=[0])
    refs_queue = [SpectrumPatternWrapper(
        graphs=pd.Series([graph_0, graph_1, graph_2]),
        losses=losses
    )] * 2
    
    # 公共参数
    base_params = {
        'loss_tolerance': 0.1,
        'mz_tolerance': 3,
        'mz_tolerance_type': 'ppm',
        'chunk_size': 5120,
        'num_workers': 2,
    }
    
    # 设备配置
    device_config = {'work_device': 'auto', 'output_device': 'auto'}
    
    # 初始化工具箱
    peak_pattern_search_toolbox = PeakPatternSearch(**{**base_params, **device_config})
    
    # 搜索测试
    results = peak_pattern_search_toolbox.run_by_queue(query_mzs_queue, refs_queue)
    
    # 结果验证
    print(results)