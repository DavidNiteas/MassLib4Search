from __future__ import annotations
from .ABCs import Searcher, SearchDataEntity, SearchConfigEntity, SearchResultsEntity
from ...utils.toolbox import PeakMZSearch
from pydantic import Field
import torch
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import dask
import dask.bag as db
from typing import Optional, Literal, List, Tuple, Union, Sequence, Hashable

class PrecursorSearchDatas(SearchDataEntity):
    
    qry_mzs: pd.Series # pd.Series[float]
    ref_mzs: pd.Series # pd.Series[float]
    qry_rts: Optional[pd.Series] = None # pd.Series[float]
    ref_rts: Optional[pd.Series] = None # pd.Series[float]
    tag_refs_ids: Optional[pd.Series] = None # pd.Series[pd.Index]
    
    def get_inputs(self):
        if self.tag_refs_ids is None:
            args = ([torch.tensor(self.qry_mzs.values)], [torch.tensor(self.ref_mzs.values)])
            kwargs = {}
            if self.qry_rts is not None and self.ref_rts is not None:
                kwargs = {
                    'qry_RTs_queue': [torch.tensor(self.qry_rts.values)],
                    'ref_RTs_queue': [torch.tensor(self.ref_rts.values)]
                }
        else:
            mzs_bag = db.from_sequence(zip(self.qry_mzs,self.tag_refs_ids))
            
            def get_mz_pair_queue(x: Tuple[float,pd.Index]) -> Tuple[torch.Tensor,torch.Tensor]:
                qry_mz, tag_refs_id = x
                ref_mzs = self.ref_mzs.loc[tag_refs_id]
                return (torch.tensor([qry_mz]), torch.tensor(ref_mzs.values))
            
            mzs_bag = mzs_bag.map(get_mz_pair_queue)
            qry_mzs_bag = mzs_bag.pluck(0)
            ref_mzs_bag = mzs_bag.pluck(1)
            args = dask.compute(qry_mzs_bag, ref_mzs_bag, scheduler='threads')
            kwargs = {}
            if self.qry_rts is not None and self.ref_rts is not None:
                rts_bag = db.from_sequence(zip(self.qry_rts,self.tag_refs_ids))
                
                def get_RT_pair_queue(x: Tuple[float,pd.Index]) -> Tuple[torch.Tensor,torch.Tensor]:
                    qry_RT, tag_refs_id = x
                    ref_RTs = self.ref_rts.loc[tag_refs_id]
                    return (torch.tensor([qry_RT]), torch.tensor(ref_RTs.values))
                
                rts_bag = rts_bag.map(get_RT_pair_queue)
                qry_rts_bag = rts_bag.pluck(0)
                ref_rts_bag = rts_bag.pluck(1)
                qry_RT_queue, ref_RT_queue = dask.compute(qry_rts_bag, ref_rts_bag, scheduler='threads')
                kwargs.update({
                    'qry_RTs_queue': qry_RT_queue,
                    'ref_RTs_queue': ref_RT_queue
                })
        return args, kwargs
        
    @classmethod
    def from_raw_data(
        cls,
        qry_mzs: Sequence[float],
        ref_mzs: Sequence[float],
        qry_rts: Optional[Sequence[float]] = None,
        ref_rts: Optional[Sequence[float]] = None,
        qry_ids: Optional[Sequence[Hashable]] = None,
        ref_ids: Optional[Sequence[Hashable]] = None,
        tag_refs_ids: Optional[Sequence[Sequence[Hashable]]] = None,
    ) -> PrecursorSearchDatas:
        qry_ids = pd.Index(qry_ids) if qry_ids is not None else pd.RangeIndex(start=0,stop=len(qry_mzs),step=1)
        ref_ids = pd.Index(ref_ids) if ref_ids is not None else pd.RangeIndex(start=0,stop=len(ref_mzs),step=1)
        qry_mzs=pd.Series(qry_mzs, index=qry_ids)
        ref_mzs=pd.Series(ref_mzs, index=ref_ids)
        qry_rts=pd.Series(qry_rts, index=qry_ids) if qry_rts is not None else None
        ref_rts=pd.Series(ref_rts, index=ref_ids) if ref_rts is not None else None
        if tag_refs_ids is not None:
            tag_refs_ids_bag = db.from_sequence(tag_refs_ids)
            tag_refs_ids_bag = tag_refs_ids_bag.map(lambda x: pd.Index(x).intersection(ref_ids))
            tag_refs_ids = dask.compute(tag_refs_ids_bag, scheduler='threads')[0]
            tag_refs_ids = pd.Series(tag_refs_ids,index=qry_ids)
        return cls(
            qry_mzs=qry_mzs,
            ref_mzs=ref_mzs,
            qry_rts=qry_rts,
            ref_rts=ref_rts,
            tag_refs_ids=tag_refs_ids
        )
    
class PrecursorSearchConfig(SearchConfigEntity):
    
    mz_tolerance: float = Field(
        default=3,
        description="质荷比偏差容忍度（单位：ppm或Da）"
    )
    mz_tolerance_type: Literal['ppm', 'Da'] = Field(
        default='ppm',
        description="质荷比容差类型：ppm（百万分之一）或 Da（道尔顿）"
    )
    RT_tolerance: float = Field(
        default=0.1,
        description="保留时间容差（单位：分钟）"
    )
    chunk_size: int = Field(
        default=5120,
        description="数据分块大小（平衡内存使用与计算效率）",
    )
    num_workers: int = Field(
        default=4,
        description="并行计算线程数",
    )
    work_device: Union[str, torch.device, Literal['auto']] = Field(
        default='auto',
        description="计算设备，自动模式（auto）优先使用CUDA可用GPU"
    )
    
    def get_inputs(self):
        return (None,None),{
            'mz_tolerance': self.mz_tolerance,
            'mz_tolerance_type': self.mz_tolerance_type,
            'RT_tolerance': self.RT_tolerance,
            'chunk_size': self.chunk_size,
            'work_device': self.work_device
        }
    
class PrecursorSearchResults(SearchResultsEntity):
    
    results_table: pd.DataFrame
    
    @classmethod
    def from_raw_results(
        cls,
        raw_results: List[Tuple[torch.Tensor, torch.Tensor]],
        data: PrecursorSearchDatas,
    ) -> PrecursorSearchResults:
        if data.tag_refs_ids is None:
            indices, delta = raw_results[0]
            qry_indices = indices[:,0].tolist()
            ref_indices = indices[:,1].tolist()
            qry_ids = data.qry_mzs.index[qry_indices].tolist()
            ref_ids = data.ref_mzs.index[ref_indices].tolist()
            results_table = pd.DataFrame({
                "qry_ids": qry_ids,
                "ref_ids": ref_ids,
                "delta": delta.tolist()
            })
        else:
            
            def mapping_ids(x: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[Hashable, pd.Index]]) -> pd.DataFrame:
                (indices, delta), (qry_id, tag_refs_id) = x
                if len(indices) == 0:
                    return pd.DataFrame()
                ref_indices = indices[:,1].tolist()
                ref_ids = tag_refs_id[ref_indices].tolist()
                qry_ids = [qry_id]*len(ref_ids)
                return pd.DataFrame({
                    "qry_ids": qry_ids,
                    "ref_ids": ref_ids,
                    "delta": delta.tolist()
                })
            
            results_table_bag = db.from_sequence(zip(raw_results, data.tag_refs_ids.items()))
            results_table_bag = results_table_bag.map(mapping_ids)
            results_table_list = dask.compute(results_table_bag, scheduler='threads')[0]
            results_table = pd.concat(results_table_list,axis=0,ignore_index=True)
        
        return cls(results_table=results_table)
    
    @property
    def qry_ids(self) -> pd.Series:
        return self.results_table['qry_ids']
    
    @property
    def ref_ids(self) -> pd.Series:
        return self.results_table['ref_ids']
    
    @property
    def deltas(self) -> NDArray[np.float32]:
        return self.results_table['delta']
    
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
        )
    
class PrecursorSearcher(Searcher):
    
    input_type = PrecursorSearchDatas
    results_type = PrecursorSearchResults
    
    config: PrecursorSearchConfig = PrecursorSearchConfig()
    
    def search_method(
        self,
        qry_ions_queue: List[torch.Tensor],
        ref_mzs_queue: List[torch.Tensor],
        qry_RTs_queue: Optional[List[torch.Tensor]] = None,
        ref_RTs_queue: Optional[List[torch.Tensor]] = None,
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        RT_tolerance: float = 0.1,
        chunk_size: int = 5120,
        num_workers: int= 4,
        work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        peak_searcher = PeakMZSearch(
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            RT_tolerance=RT_tolerance,
            adduct_co_occurrence_threshold=1,
            chunk_size=chunk_size,
            num_workers=num_workers,
            work_device=work_device,
            output_device='cpu'
        )
        raw_results = peak_searcher.run_by_queue(qry_ions_queue, ref_mzs_queue, qry_RTs_queue, ref_RTs_queue)
        return raw_results
    
    def run(self, data: PrecursorSearchDatas) -> PrecursorSearchResults:
        return super().run(data)