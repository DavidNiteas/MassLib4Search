import pytest
import numpy as np
import pandas as pd
import torch
from masslib4search.snaps.MassSearchTools.txn.searcher.embedding_searcher import (
    EmbeddingSearchDatas,
    EmbeddingSearchConfig,
    EmbeddingSearchResults,
    EmbeddingSearcher,
)

def embedding_searcher() -> EmbeddingSearchResults:
    # 测试数据
    qry_embs = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    ref_embs = [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [7.1, 8.1, 9.1]
    ]
    qry_ids = [1,2,3]
    ref_ids = [1,2,3,4,5]
    tag_refs_ids = [[1,2],[3],[4,5]]

    inputs = EmbeddingSearchDatas.from_raw_data(qry_embs=qry_embs, ref_embs=ref_embs, qry_ids=qry_ids, ref_ids=ref_ids, tag_refs_ids=tag_refs_ids)

    # 测试配置
    config = EmbeddingSearchConfig(
        chunk_size=10000,
        work_device='auto'
    )

    # 初始化搜索器
    searcher = EmbeddingSearcher(config=config)

    # 运行搜索
    results = searcher.run(inputs)

    return results

@pytest.mark.MassSearchTools_embedding_searcher
def test_embedding_searcher():
    
    results = embedding_searcher()

    # 验证结果
    assert isinstance(results, EmbeddingSearchResults)
    assert len(results.results_table) == 5
    assert len(results.tag_refs_ids) == 3
    assert np.allclose(results.qry_ids.values,[1,1,2,3,3])
    assert np.allclose(results.ref_ids.values,[1,2,3,4,5])
    assert np.allclose(results.scores, [1.0,0.999859,1.0,1.0,1.0], atol=1e-5)
    assert np.allclose(results.unique_qry_ids.values, [1, 2, 3])
    assert np.allclose(results.unique_ref_ids.values, [1, 2, 3, 4, 5])
    assert np.allclose(results.tag_refs_ids.loc[1].values, [1, 2])
    assert np.allclose(results.tag_refs_ids.loc[2].values, [3])
    assert np.allclose(results.tag_refs_ids.loc[3].values, [4, 5])

if __name__ == '__main__':
    results = embedding_searcher()
    print(results.results_table)
    print(results.tag_refs_ids)
