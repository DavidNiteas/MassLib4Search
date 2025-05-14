import pytest
from masslib4search.snaps.MassSearchTools.txn.searcher.precursor_searcher import (
    PrecursorSearchDatas,
    PrecursorSearchConfig,
    PrecursorSearchResults,
    PrecursorSearcher,
)
import numpy as np

def precursor_searcher() -> PrecursorSearchResults:
    
    # 测试数据
    inputs = PrecursorSearchDatas.from_raw_data(
        qry_mzs=[100, 200, 300],
        ref_mzs=[100, 200, 300, 200, 100],
        qry_ids=[1, 2, 3],
        ref_ids=[1, 2, 3, 4, 5],
        tag_refs_ids=[[1, 2, 3], [2, 3, 4], [4, 5]]
    )
    
    # 测试配置
    config = PrecursorSearchConfig(
        mz_tolerance=10,
        mz_tolerance_type='ppm',
        RT_tolerance=0.1,
        chunk_size=10000,
        work_device='auto'
    )
    
    # 初始化搜索器
    searcher = PrecursorSearcher(config=config)
    
    # 运行搜索
    results = searcher.run(inputs)
    
    return results

@pytest.mark.MassSearchTools_precursor_searcher
def test_precursor_searcher():
    
    # 运行搜索
    results = precursor_searcher()
    
    # 验证结果
    assert isinstance(results, PrecursorSearchResults)
    assert len(results.results_table) == 3
    assert len(results.tag_refs_ids) == 2
    assert np.allclose(results.qry_ids.values,[1,2,2])
    assert np.allclose(results.ref_ids.values,[1,2,4])
    assert np.allclose(results.deltas, [0, 0, 0])
    assert np.allclose(results.unique_qry_ids.values, [1, 2])
    assert np.allclose(results.unique_ref_ids.values, [1, 2, 4])
    assert np.allclose(results.tag_refs_ids.loc[1].values, [1])
    assert np.allclose(results.tag_refs_ids.loc[2].values, [2, 4])
    
if __name__ == '__main__':
    results = precursor_searcher()
    print(results.results_table)
    print(results.tag_refs_ids)