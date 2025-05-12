import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumSimilarity
from masslib4search.snaps.MassSearchTools.utils.similarity.operators import SpectramSimilarityOperator, MSEntropyOperator
import torch

@pytest.mark.MassSearchTools_utils_similarity
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'}
])
def test_spectrum_similarity(query_spec_queue, ref_spec_queue, device_config):
    
    # SpectrumSimilarity 公共参数
    similarity_base_params = {
        "sim_operator": MSEntropyOperator,
        "num_cuda_workers": 4,
        "num_dask_workers": 4,
        "dask_mode": "threads"
    }
    
    # 初始化SpectrumSimilarity工具箱
    similarity_toolbox = SpectrumSimilarity(
        work_device=device_config['work_device'],
        output_device=device_config['output_device'],
        **similarity_base_params
    )
    
    # 获取SpectrumSimilarity
    similarity_matrix_queue = similarity_toolbox.run_by_queue(query_spec_queue, ref_spec_queue)
    
    # 断言
    assert len(similarity_matrix_queue) == 2
    assert similarity_matrix_queue[0].shape == (len(query_spec_queue[0]), len(ref_spec_queue[0]))
    assert similarity_matrix_queue[1].shape == (len(query_spec_queue[1]), len(ref_spec_queue[1]))
    assert torch.allclose(similarity_matrix_queue[0][0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(similarity_matrix_queue[1][0, 0], torch.tensor(1.0), atol=1e-6)
