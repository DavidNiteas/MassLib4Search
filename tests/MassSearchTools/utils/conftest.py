import pytest
import torch
from typing import List

@pytest.fixture(scope='session')
def query_spec() -> List[torch.Tensor]:
    return [
        torch.tensor([
            [149.05970595,0.5],
            [277.15869171,0.3],
            [295.1692564,0.4],
            [313.17982108,0.6],
            [373.20095045,0.7],
            [433.22207982,0.8],
            [581.27450931,1.0],
        ]),
    ] * 2
    
@pytest.fixture(scope='session')
def ref_spec() -> List[torch.Tensor]:
    return [
        torch.tensor([ # 与query完全一样
            [149.05970595,0.5],
            [277.15869171,0.3],
            [295.1692564,0.4],
            [313.17982108,0.6],
            [373.20095045,0.7],
            [433.22207982,0.8],
            [581.27450931,1.0],
        ]),
        torch.tensor([ # 与query仅存在强度差异
            [149.05970595,0.2],
            [277.15869171,0.3],
            [295.1692564,0.7],
            [313.17982108,0.3],
            [373.20095045,0.5],
            [433.22207982,0.9],
            [581.27450931,1.0],
        ]),
        torch.tensor([ # 与query存在mz和intens的差异，但长度一致
            [149.05970595,0.2],
            [237.15869171,0.3],
            [295.1692564,0.7],
            [303.17982108,0.3],
            [363.20095045,0.5],
            [423.22207982,0.9],
            [581.27450931,1.0],
        ]),
        torch.tensor([ # 与query完全不一致
            [231.15869171,0.3],
            [290.1692564,0.7],
            [303.17982108,0.3],
            [363.20095045,0.5],
            [423.22207982,1.0],
        ]),
    ]
    
@pytest.fixture(scope='session')
def query_spec_queue(query_spec) -> List[List[torch.Tensor]]:
    return [query_spec] * 2

@pytest.fixture(scope='session')
def ref_spec_queue(ref_spec) -> List[List[torch.Tensor]]:
    return [ref_spec] * 2