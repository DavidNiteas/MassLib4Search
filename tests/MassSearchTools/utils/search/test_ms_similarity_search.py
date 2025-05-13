import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumSimilaritySearch
from masslib4search.snaps.MassSearchTools.utils.similarity.operators import MSEntropyOperator
import torch

@pytest.mark.MassSearchTools_utils_search
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'}
])
def test_spectrum_similarity_search(query_spec_queue, ref_spec_queue, device_config):
    # SpectrumSimilaritySearch 公共参数
    search_base_params = {
        "sim_operator": MSEntropyOperator,
        "top_k": 2, 
        "num_cuda_workers": 4,
        "num_dask_workers": 4,
        "dask_mode": "threads"
    }
    
    # 初始化SpectrumSimilaritySearch工具箱
    search_toolbox = SpectrumSimilaritySearch(
        work_device=device_config['work_device'],
        output_device=device_config['output_device'],
        **search_base_params
    )
    
    # 执行谱图相似度搜索
    search_results = search_toolbox.run_by_queue(query_spec_queue, ref_spec_queue)
    
    # 断言
    assert len(search_results) == 2
    assert torch.allclose(search_results[0][0], search_results[1][0])
    assert torch.allclose(search_results[0][1], search_results[1][1])
    assert search_results[0][0].shape == (2, search_base_params['top_k'])
    assert search_results[0][1].shape == (2, search_base_params['top_k'])
    assert torch.all(torch.diff(search_results[0][1]) <= 0).item() == True
    assert torch.allclose(search_results[0][0][0,0],torch.tensor(0,dtype=torch.long))
    assert torch.allclose(search_results[0][1][0,0],torch.tensor(1.0,dtype=torch.float32))
