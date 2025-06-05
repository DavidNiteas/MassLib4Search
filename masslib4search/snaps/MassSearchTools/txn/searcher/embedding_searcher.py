from __future__ import annotations
from .ABCs import Searcher, SearchDataEntity, SearchConfigEntity, SearchResultsEntity
from ...utils.toolbox import EmbeddingSimilaritySearch, CosineOperator, EmbbedingSimilarityOperator
from pydantic import Field
import torch
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike,NDArray
import dask
import dask.bag as db
from typing import Optional, Literal, List, Tuple, Union, Sequence, Hashable, Type

class EmbeddingSearchDatas(SearchDataEntity):
    
    qry_embs: pd.Series # pd.Series[torch.Tensor]
    ref_embs: pd.Series # pd.Series[torch.Tensor]
    tag_refs_ids: Optional[pd.Series] = None # pd.Series[pd.Index]
    
    def get_inputs(self):
        if self.tag_refs_ids is None:
            args = (
                [torch.stack(self.qry_embs.tolist())], 
                [torch.stack(self.ref_embs.tolist())]
            )
        else:
            embs_bag = db.from_sequence(zip(self.qry_embs,self.tag_refs_ids))
            
            def get_pair_queue(x: Tuple[torch.Tensor,pd.Index]) -> Tuple[torch.Tensor,torch.Tensor]:
                qry_emb, tag_refs_id = x
                ref_embs = self.ref_embs.loc[tag_refs_id].tolist()
                if len(ref_embs) == 0:
                    ref_embs = torch.tensor([],dtype=self.ref_embs.iloc[0].dtype).reshape(0,self.ref_embs.iloc[0].shape[-1])
                else:
                    ref_embs = torch.stack(ref_embs)
                return (qry_emb.reshape(1,-1), ref_embs)
            
            embs_bag = embs_bag.map(get_pair_queue)
            qry_embs_bag = embs_bag.pluck(0)
            ref_embs_bag = embs_bag.pluck(1)
            args = dask.compute(qry_embs_bag, ref_embs_bag, scheduler='threads')
        return args, {}
    
    @classmethod
    def from_raw_data(
        cls,
        qry_embs: ArrayLike,
        ref_embs: ArrayLike,
        qry_ids: Optional[Sequence[Hashable]] = None,
        ref_ids: Optional[Sequence[Hashable]] = None,
        tag_refs_ids: Optional[Sequence[Sequence[Hashable]]] = None,
    ) -> EmbeddingSearchDatas:
        qry_ids = pd.Index(qry_ids) if qry_ids is not None else pd.RangeIndex(start=0,stop=len(qry_embs),step=1)
        ref_ids = pd.Index(ref_ids) if ref_ids is not None else pd.RangeIndex(start=0,stop=len(ref_embs),step=1)
        qry_embs_bag = db.from_sequence(qry_embs,npartitions=1)
        ref_embs_bag = db.from_sequence(ref_embs,npartitions=1)
        qry_embs_bag = qry_embs_bag.map(lambda x: torch.tensor(x,dtype=torch.float32))
        ref_embs_bag = ref_embs_bag.map(lambda x: torch.tensor(x,dtype=torch.float32))
        if tag_refs_ids is not None:
            tag_refs_ids_bag = db.from_sequence(tag_refs_ids)
            tag_refs_ids_bag = tag_refs_ids_bag.map(lambda x: pd.Index(x).intersection(ref_ids))
        else:
            tag_refs_ids_bag = None
        qry_embs, ref_embs, tag_refs_ids = dask.compute(qry_embs_bag, ref_embs_bag, tag_refs_ids_bag, scheduler='threads')
        qry_embs = pd.Series(qry_embs,index=qry_ids)
        ref_embs = pd.Series(ref_embs,index=ref_ids)
        if tag_refs_ids is not None:
            tag_refs_ids = pd.Series(tag_refs_ids,index=qry_ids)
        return cls(
            qry_embs=qry_embs, 
            ref_embs=ref_embs, 
            tag_refs_ids=tag_refs_ids,
        )
        
class EmbeddingSearchConfig(SearchConfigEntity):
    
    sim_operator: EmbbedingSimilarityOperator = Field(
        CosineOperator,
        description='向量相似度计算算子（默认使用CosineOperator）'
    )
    chunk_size: int = Field(
        5120,
        description='进行向量相似度计算时最小的计算单元大小'
    )
    top_k: Optional[int] = Field(
        None,
        description='返回的结果中最多返回的结果数（默认返回所有结果）'
    )
    batch_size: int = Field(
        128,
        description='并行计算时每个批次的大小'
    )
    num_workers: int = Field(
        4,
        description='并行计算使用的工作线程数'
    )
    work_device: Union[str, torch.device, Literal['auto']] = Field(
        'auto',
        description='向量相似度计算使用的计算设备'
    )
    operator_kwargs: Optional[dict] = Field(
        None,
        description='传入向量相似度计算算子的其它参数'
    )
    
    def get_inputs(self):
        return (None,None),{
            'sim_operator': self.sim_operator,
            'chunk_size': self.chunk_size,
            'top_k': self.top_k,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'work_device': self.work_device,
            'operator_kwargs': self.operator_kwargs,
        }
        
class EmbeddingSearchResults(SearchResultsEntity):
    
    results_table: pd.DataFrame
    
    @classmethod
    def from_raw_results(
        cls,
        raw_results: List[Tuple[torch.Tensor,torch.Tensor]],
        data: EmbeddingSearchDatas,
    ) -> EmbeddingSearchResults:
        if data.tag_refs_ids is None:
            indices, scores = raw_results[0]
            qry_indices = indices[:,0].tolist()
            ref_indices = indices[:,1].tolist()
            qry_ids = data.qry_embs.index[qry_indices].tolist()
            ref_ids = data.ref_embs.index[ref_indices].tolist()
            topk = (indices[:,2]+1).tolist()
            results_table = pd.DataFrame({
                'qry_ids': qry_ids,
                'ref_ids': ref_ids,
                'topk': topk,
                'scores': scores.tolist(),
            })
        else:
            
            def mapping_ids(x: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[Hashable, pd.Index]]) -> pd.DataFrame:
                (indices, scores), (qry_id, tag_refs_id) = x
                if len(indices) == 0:
                    return pd.DataFrame()
                ref_indices = indices[:,1].tolist()
                ref_ids = tag_refs_id[ref_indices].tolist()
                qry_ids = [qry_id]*len(ref_ids)
                topk = (indices[:,2]+1).tolist()
                return pd.DataFrame({
                    'qry_ids': qry_ids,
                    'ref_ids': ref_ids,
                    'topk': topk,
                    'scores': scores.tolist(),
                })
            
            results_bag = db.from_sequence(zip(raw_results, data.tag_refs_ids.items()))
            results_bag = results_bag.map(mapping_ids)
            results_table = dask.compute(results_bag, scheduler='threads')[0]
            results_table = pd.concat(results_table, axis=0, ignore_index=True)
            
        return cls(results_table=results_table)
    
    @property
    def qry_ids(self) -> pd.Series:
        return self.results_table['qry_ids']
    
    @property
    def ref_ids(self) -> pd.Series:
        return self.results_table['ref_ids']
    
    @property
    def topk(self) -> pd.Series:
        return self.results_table['topk']
    
    @property
    def scores(self) -> NDArray[np.float32]:
        return self.results_table['scores'].values
    
    @property
    def unique_qry_ids(self) -> pd.Index:
        return pd.Index(self.qry_ids.unique())
    
    @property
    def unique_ref_ids(self) -> pd.Index:
        return pd.Index(self.ref_ids.unique())
    
    @property
    def tag_refs_ids(self) -> pd.Series:
        return (
            self.results_table
            .groupby('qry_ids', group_keys=False)['ref_ids']
            .apply(pd.Index)
            .apply(lambda x: x.unique())
        )
    
class EmbeddingSearcher(Searcher):
    
    input_type = EmbeddingSearchDatas
    results_type = EmbeddingSearchResults
    
    config: EmbeddingSearchConfig = EmbeddingSearchConfig()
    
    def search_method(
        self,
        qry_embs_queue: List[torch.Tensor],
        ref_embs_queue: List[torch.Tensor],
        sim_operator: Type[EmbbedingSimilarityOperator] = CosineOperator,
        chunk_size: int = 5120,
        top_k: Optional[int] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        work_device: Union[str, torch.device, Literal['auto']] = 'auto',
        operator_kwargs: Optional[dict] = None,
    ) -> List[Tuple[torch.Tensor,torch.Tensor]]:
        emb_simi_searcher = EmbeddingSimilaritySearch(
            sim_operator=sim_operator,
            chunk_size=chunk_size,
            top_k=top_k,
            batch_size=batch_size,
            num_workers=num_workers,
            work_device=work_device,
            output_device='cpu',
            operator_kwargs=operator_kwargs,
            output_mode='hit',
        )
        results = emb_simi_searcher.run_by_queue(qry_embs_queue, ref_embs_queue)
        return results
    
    def run(self, data: EmbeddingSearchDatas) -> EmbeddingSearchResults:
        return super().run(data)