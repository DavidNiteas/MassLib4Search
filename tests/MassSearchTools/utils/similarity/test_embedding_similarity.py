import pytest
from masslib4search.snaps.MassSearchTools.utils.similarity import emb_similarity_by_queue
from masslib4search.snaps.MassSearchTools.utils.embedding import binning
from masslib4search.snaps.MassSearchTools.utils.spectrum_tools import split_spectrum_by_queue

@pytest.mark.MassSearchTools_utils_similarity
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'cpu'}
])
def test_emb_similarity_by_queue(query_spec_queue, ref_spec_queue, device_config):
    
    def binning_wrapper(spec_queue,device_config):
    
        # binning公共参数
        base_params = {
            "binning_window": (50.0, 1000.0, 1.0),
            "pool_method": "sum",
            "batch_size": 128,
            "num_workers": 4
        }
    
        # 执行带设备参数的binning
        binned_spectrum = binning(
            spec_queue,
            **{**base_params, **device_config}  # 合并参数
        )
    
        return binned_spectrum
    
    