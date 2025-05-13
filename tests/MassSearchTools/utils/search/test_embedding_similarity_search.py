import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumBinning, EmbeddingSimilaritySearch
from masslib4search.snaps.MassSearchTools.utils.similarity.operators import CosineOperator
import torch

@pytest.mark.MassSearchTools_utils_search
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'cpu'}
])
def test_embedding_similarity_search(query_spec_queue, ref_spec_queue, device_config):
    # Binning 公共参数
    binning_base_params = {
        "binning_window": (50.0, 1000.0, 1.0),
        "pool_method": "sum",
        "batch_size": 128,
        "num_workers": 4
    }
    
    # 初始化Binning工具箱
    binning_toolbox = SpectrumBinning(
        **{**binning_base_params, **device_config}  # 合并参数
    )
    
    # 获取Binning-Embedding
    binned_query_spec = binning_toolbox.run_by_queue(query_spec_queue)
    binned_ref_spec = binning_toolbox.run_by_queue(ref_spec_queue)
    
    # EmbeddingSimilaritySearch 公共参数
    similarity_search_base_params = {
        "sim_operator": CosineOperator,
        "chunk_size": 1024,
        "batch_size": 128,
        "num_workers": 4,
        "top_k": 2,  # 假设我们只返回最相关的前2个向量
    }
    
    # 初始化EmbeddingSimilaritySearch工具箱
    similarity_search_toolbox = EmbeddingSimilaritySearch(
        work_device=device_config['work_device'],
        output_device=device_config['output_device'],
        **similarity_search_base_params
    )
    
    # 获取相似度搜索结果
    search_results = similarity_search_toolbox.run_by_queue(binned_query_spec, binned_ref_spec)
    
    # 验证结果
    assert len(search_results) == 2  # 假设输入的查询队列有两个元素
    assert torch.allclose(search_results[0][0], search_results[1][0])
    assert torch.allclose(search_results[0][1], search_results[1][1])
    assert search_results[0][0].shape == (2, similarity_search_base_params['top_k'])
    assert search_results[0][1].shape == (2, similarity_search_base_params['top_k'])
    assert torch.all(torch.diff(search_results[0][1]) <= 0).item() == True
    assert torch.allclose(search_results[0][0][0,0],torch.tensor(0,dtype=torch.long))
    assert torch.allclose(search_results[0][1][0,0],torch.tensor(1.0,dtype=torch.float32))