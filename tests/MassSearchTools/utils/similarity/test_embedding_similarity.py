import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumBinning, EmbeddingSimilarity
from masslib4search.snaps.MassSearchTools.utils.similarity.operators import CosineOperator
import torch

@pytest.mark.MassSearchTools_utils_similarity
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'cpu'}
])
def test_embedding_similarity(query_spec_queue, ref_spec_queue, device_config):
    
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
    
    # EmbeddingSimilarity 公共参数
    similarity_base_params = {
        "sim_operator": CosineOperator,
        "chunk_size": 1024,
        "batch_size": 128,
        "num_workers": 4
    }
    
    # 初始化EmbeddingSimilarity工具箱
    similarity_toolbox = EmbeddingSimilarity(
        work_device="auto",
        output_device=device_config['output_device'],
        **similarity_base_params
    )
    
    # 获取EmbeddingSimilarity
    similarity_matrix_queue = similarity_toolbox.run_by_queue(binned_query_spec, binned_ref_spec)
    
    # 断言
    assert len(similarity_matrix_queue) == 2
    assert torch.allclose(similarity_matrix_queue[0], similarity_matrix_queue[1], atol=1e-6)
    assert similarity_matrix_queue[0].shape == (2,4)
    assert torch.allclose(similarity_matrix_queue[0][0,0], torch.tensor(1.0), atol=1e-6)