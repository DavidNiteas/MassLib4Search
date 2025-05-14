import pytest
from masslib4search.snaps.MassSearchTools.txn.searcher.fragment_searcher import (
    FragmentSearcher,
    FragmentSearchConfig,
    FragmentSearchDatas,
    FragmentSearchResults
)
import numpy as np

def fragment_searcher() -> FragmentSearchResults:
    
    # 测试数据
    inputs = FragmentSearchDatas.from_raw_data(
        qry_mzs=[100, 200, 300],
        ref_mzs=[
            [100,101,102],
            [199,200,201],
            [298,299,300],
            [100,200,300],
        ],
        adducts=["[M]+", "[M+H]+", "[M+2H]+"],
        qry_ids=[1, 2, 3],
        ref_ids=[1, 2, 3, 4],
        tag_refs_ids=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3]]
    )
    
    # 测试配置
    config = FragmentSearchConfig(
        mz_tolerance=10,
        mz_tolerance_type='ppm',
        RT_tolerance=0.1,
        chunk_size=10000,
        work_device='auto',
        adduct_co_occurrence_threshold=1
    )
    
    # 初始化搜索器
    searcher = FragmentSearcher(config=config)
    
    # 运行搜索
    results = searcher.run(inputs)
    
    return results

@pytest.mark.MassSearchTools_fragment_searcher
def test_fragment_searcher():
    
    # 运行测试
    results = fragment_searcher()
    
    # 验证结果
    assert isinstance(results, FragmentSearchResults)
    assert len(results.results_table) == 5
    assert len(results.tag_refs_ids) == 3
    assert np.allclose(results.qry_ids.values,[1,1,2,2,3])
    assert np.allclose(results.ref_ids.values,[1,4,2,4,3])
    assert np.allclose(results.deltas, [0, 0, 0, 0, 0])
    assert np.all(results.adducts == ['[M]+', '[M]+', '[M+H]+', '[M+H]+', '[M+2H]+'])
    assert np.allclose(results.unique_qry_ids.values, [1, 2, 3])
    assert np.allclose(results.unique_ref_ids.values, [1, 4, 2, 3])
    assert np.allclose(results.tag_refs_ids.loc[1].values, [1, 4])
    assert np.allclose(results.tag_refs_ids.loc[2].values, [2, 4])
    assert np.allclose(results.tag_refs_ids.loc[3].values, [3])

if __name__ == '__main__':
    results = fragment_searcher()
    print(results.results_table)
    print(results.unique_ref_ids)
    print(results.tag_refs_ids)