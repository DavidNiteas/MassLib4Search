from __future__ import annotations
from .ABCs import Searcher
from .precursor_searcher import PrecursorSearchDatas,PrecursorSearchConfig,PrecursorSearchResults
from ...utils.toolbox import PeakMZSearch
from pydantic import Field
import torch
import pandas as pd
import dask
import dask.bag as db
from typing import Optional, Literal, List, Tuple, Union, Sequence, Hashable

class FragmentSearchDatas(PrecursorSearchDatas):
    
    ref_mzs: pd.DataFrame
    
    @classmethod
    def from_raw_data(
        cls,
        qry_mzs: Sequence[float],
        ref_mzs: Sequence[Sequence[float]],
        adducts: Sequence[Hashable],
        qry_rts: Optional[Sequence[float]] = None,
        ref_rts: Optional[Sequence[float]] = None,
        qry_ids: Optional[Sequence[Hashable]] = None,
        ref_ids: Optional[Sequence[Hashable]] = None,
        tag_refs_ids: Optional[Sequence[Sequence[Hashable]]] = None,
    ) -> FragmentSearchDatas:
        qry_ids = pd.Index(qry_ids) if qry_ids is not None else pd.RangeIndex(start=0,stop=len(qry_mzs),step=1)
        ref_ids = pd.Index(ref_ids) if ref_ids is not None else pd.RangeIndex(start=0,stop=len(ref_mzs),step=1)
        qry_mzs=pd.Series(qry_mzs, index=qry_ids)
        ref_mzs=pd.DataFrame(ref_mzs, index=ref_ids, columns=adducts)
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
        
class FragmentSearchConfig(PrecursorSearchConfig):
    
    adduct_co_occurrence_threshold: int = Field(
        default=1,
        ge=1,
        description='加合物共现过滤阈值，单个ref的中必须存在阈值以上的加合物匹配才会被算作命中'
    )
    
    def get_inputs(self):
        args, kwargs = super().get_inputs()
        kwargs['adduct_co_occurrence_threshold'] = self.adduct_co_occurrence_threshold
        return args, kwargs
    
class FragmentSearchResults(PrecursorSearchResults):
    
    @classmethod
    def from_raw_results(
        cls,
        raw_results: List[Tuple[torch.Tensor, torch.Tensor]],
        data: FragmentSearchDatas,
    ) -> FragmentSearchResults:
        if data.tag_refs_ids is None:
            indices, delta = raw_results[0]
            qry_indices = indices[:,0].tolist()
            ref_indices = indices[:,1].tolist()
            adduct_indices = indices[:,2].tolist()
            qry_ids = data.qry_mzs.index[qry_indices].tolist()
            ref_ids = data.ref_mzs.index[ref_indices].tolist()
            adducts = data.ref_mzs.columns[adduct_indices].tolist()
            results_table = pd.DataFrame({
                "qry_ids": qry_ids,
                "ref_ids": ref_ids,
                "adducts": adducts,
                "delta": delta.tolist()
            })
        else:
            
            def mapping_ids(x: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[Hashable, pd.Index]]) -> pd.DataFrame:
                (indices, delta), (qry_id, tag_refs_id) = x
                if len(indices) == 0:
                    return pd.DataFrame()
                ref_indices = indices[:,1].tolist()
                adduct_indices = indices[:,2].tolist()
                ref_ids = tag_refs_id[ref_indices].tolist()
                qry_ids = [qry_id]*len(ref_ids)
                adducts = data.ref_mzs.columns[adduct_indices].tolist()
                return pd.DataFrame({
                    "qry_ids": qry_ids,
                    "ref_ids": ref_ids,
                    "adducts": adducts,
                    "delta": delta.tolist()
                })
            results_table_bag = db.from_sequence(zip(raw_results, data.tag_refs_ids.items()))
            results_table_bag = results_table_bag.map(mapping_ids)
            results_table_list = dask.compute(results_table_bag, scheduler='threads')[0]
            results_table = pd.concat(results_table_list,axis=0,ignore_index=True)
        
        return cls(results_table=results_table)
    
    @property
    def adducts(self) -> pd.Series:
        return self.results_table['adducts']
    
class FragmentSearcher(Searcher):
    
    input_type = FragmentSearchDatas
    results_type = FragmentSearchResults
    
    config: FragmentSearchConfig = FragmentSearchConfig()
    
    def search_method(
        self,
        qry_ions_queue: List[torch.Tensor],
        ref_mzs_queue: List[torch.Tensor],
        qry_RTs_queue: Optional[List[torch.Tensor]] = None,
        ref_RTs_queue: Optional[List[torch.Tensor]] = None,
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1,
        chunk_size: int = 5120,
        work_device: Union[str, torch.device, Literal['auto']] = 'auto',
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        peak_searcher = PeakMZSearch(
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            RT_tolerance=RT_tolerance,
            adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
            chunk_size=chunk_size,
            work_device=work_device,
            output_device='cpu'
        )
        raw_results = peak_searcher.run_by_queue(qry_ions_queue, ref_mzs_queue, qry_RTs_queue, ref_RTs_queue)
        return raw_results
    
    def run(self, data: FragmentSearchDatas) -> FragmentSearchResults:
        return super(FragmentSearcher, self).run(data)