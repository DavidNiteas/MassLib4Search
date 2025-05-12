import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import PeakMZSearch
import torch

@pytest.mark.MassSearchTools_utils_search
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'cpu'}
])
def test_peak_mz_search(device_config):
    
    # 测试数据
    query_mzs_queue = [torch.tensor([149.05970595, 277.15869171, 295.1692564, 313.17982108, 373.20095045, 433.22207982, 581.27450931])] * 2
    ref_mzs_queue = [torch.tensor([149.05970595, 237.15869171, 295.1692564, 303.17982108, 363.20095045, 423.22207982, 581.27450931])] * 2
    
    # 公共参数
    base_params = {
        "mz_tolerance": 5,
        "mz_tolerance_type": "ppm",
        "RT_tolerance": 0.1,
        "adduct_co_occurrence_threshold":1,
        "chunk_size":5120,
    }
    
    # 初始化工具箱
    peak_search_toolbox = PeakMZSearch(
        **{**base_params, **device_config}
    )
    
    # 搜索测试
    search_results = peak_search_toolbox.run_by_queue(query_mzs_queue,ref_mzs_queue)
    
    # 结果断言
    assert len(search_results) == 2
    assert torch.allclose(search_results[0][0], search_results[1][0],atol=1e-6)
    assert torch.allclose(search_results[0][1], search_results[1][1],atol=1e-6)
    assert len(search_results[0][0]) == len(search_results[0][1]) == 3
    assert torch.allclose(search_results[0][0],torch.tensor([[0,0],[2,2],[6,6]]))
    
if __name__ == '__main__':
    peak_search_toolbox = PeakMZSearch(
        **{"mz_tolerance": 5,
        "mz_tolerance_type": "ppm",
        "RT_tolerance": 0.1,
        "adduct_co_occurrence_threshold":1,
        "chunk_size":5120,
        "work_device": "auto",
        "output_device": "auto"}
    )
    query_mzs_queue = [torch.tensor([149.05970595, 277.15869171, 295.1692564, 313.17982108, 373.20095045, 433.22207982, 581.27450931])] * 2
    ref_mzs_queue = [torch.tensor([149.05970595, 237.15869171, 295.1692564, 303.17982108, 363.20095045, 423.22207982, 581.27450931])] * 2
    search_results = peak_search_toolbox.run_by_queue(query_mzs_queue,ref_mzs_queue)
    print(search_results)