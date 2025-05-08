from .ABCs import Searcher, SearchDataEntity, SearchConfigEntity, SearchResultsEntity
from ...utils.search.ms_peak_search import mz_search_by_queue
import torch
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import logging
from typing import Optional, Literal, List, Tuple, Union, Sequence

class PrecursorSearchDatas(SearchDataEntity):
    
    __slots__ = SearchDataEntity.__slots__ + ['query_ions_mz', 'query_ions_id','ref_ions_mz','ref_ions_id']
    qry_ions_queue: List[torch.Tensor]
    ref_mzs_queue: List[torch.Tensor]
    query_RTs_queue: Optional[List[Optional[torch.Tensor]]] = None
    ref_RTs_queue: Optional[List[Optional[torch.Tensor]]] = None
    
    def get_inputs(self):
        return (
            (self.qry_ions_queue, self.ref_mzs_queue),
            {
                'query_RTs_queue': self.query_RTs_queue,
                'ref_RTs_queue': self.ref_RTs_queue,
            }
        )
        
    