import pytest
from masslib4search.snaps.MassSearchTools.utils.toolbox import SpectrumBinning

@pytest.mark.MassSearchTools_utils_embedding
@pytest.mark.parametrize("device_config", [
    {'work_device': 'auto', 'output_device': 'auto'},
    {'work_device': 'cpu', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'auto'},
    {'work_device': 'cuda:0', 'output_device': 'cpu'}
])
def test_binning(query_spec_queue, device_config):
    
    # 公共参数
    base_params = {
        "binning_window": (50.0, 1000.0, 1.0),
        "pool_method": "sum",
        "batch_size": 128,
        "num_workers": 4
    }
    
    # 初始化工具箱
    binning_toolbox = SpectrumBinning(
        **{**base_params, **device_config}  # 合并参数
    )
    binning_toolbox.run_by_queue(query_spec_queue)