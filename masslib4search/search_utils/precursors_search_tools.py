import torch
from .base_search_tools import (
    broadcast,get_delta_matrix,ppm_convert,
    get_bool_matrix,get_indices
)
import pandas as pd
import numpy as np
import dask.bag as db
import logging
from numpy.typing import NDArray
from typing import Optional, Literal, List, Tuple

logger = logging.getLogger(__name__)

@torch.no_grad()
def get_precursor_hits(
    qry_ions_array: torch.Tensor,          # shape: (n_ions,) float32
    ref_precursor_mzs_array: torch.Tensor, # shape: (n_ref,) float32
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[torch.Tensor] = None, # shape: (n_ref,) float16
    query_RTs: Optional[torch.Tensor] = None, # shape: (n_ions,) float16
    RT_tolerance: float = 0.1,
) -> torch.Tensor:
    
    Q, R = broadcast(qry_ions_array.unsqueeze(1), ref_precursor_mzs_array.unsqueeze(0))
    D = get_delta_matrix(Q, R)
    if mz_tolerance_type == 'ppm':
        D = ppm_convert(D, R)
    B = get_bool_matrix(D, mz_tolerance)
    if ref_RTs is not None and query_RTs is not None:
        RT_Q, RT_R = broadcast(query_RTs.unsqueeze(1), ref_RTs.unsqueeze(0))
        RT_D = get_delta_matrix(RT_Q, RT_R)
        RT_B = get_bool_matrix(RT_D, RT_tolerance)
        B = B & RT_B
    I = get_indices(B)
    return I

@torch.no_grad()
def search_block_precursors(
    qry_ions_array: torch.Tensor,
    ref_precursor_mzs_array: torch.Tensor,
    qry_index: pd.Index,
    ref_index: pd.Index,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[torch.Tensor] = None,
    query_RTs: Optional[torch.Tensor] = None,
    RT_tolerance: float = 0.1,
) -> pd.Series:
    I = get_precursor_hits(
        qry_ions_array, ref_precursor_mzs_array,
        mz_tolerance, mz_tolerance_type,
        ref_RTs,query_RTs,RT_tolerance
    ).cpu()
    hitted_qry = qry_index[I[:, 0]]
    hitted_ref = ref_index[I[:, 1]]
    return pd.Series(hitted_ref, index=hitted_qry)

def search_precursors(
    qry_ions_queue: List[pd.Series], # shape: (n_ions,) float32
    ref_precursor_mzs_queue: List[pd.Series], # shape: (n_ref,) float32
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[NDArray[np.float_]] = None,
    query_RTs: Optional[NDArray[np.float_]] = None,
    RT_tolerance: float = 0.1,
    device: str = 'cuda',
    num_workers: int = 4,
) -> pd.Series:
    
    logger.info('Initializing precursor search...')
    qry_bag = db.from_sequence(qry_ions_queue)
    ref_bag = db.from_sequence(ref_precursor_mzs_queue)
    product_bag = qry_bag.product(ref_bag)

    def search_block_in_bag(chunk: Tuple[pd.Series, pd.Series]) -> pd.Series:
        qry_series, ref_series = chunk
        qry_array = torch.as_tensor(qry_series.values, dtype=torch.float32, device=device)
        ref_array = torch.as_tensor(ref_series.values, dtype=torch.float32, device=device)
        qry_chunk_RTs = torch.as_tensor(query_RTs[qry_series.index], dtype=torch.float16, device=device) if query_RTs is not None else None
        ref_chunk_RTs = torch.as_tensor(ref_RTs[ref_series.index], dtype=torch.float16, device=device) if ref_RTs is not None else None
        return search_block_precursors(
            qry_array, ref_array,
            qry_series.index, ref_series.index,
            mz_tolerance, mz_tolerance_type,
            ref_chunk_RTs, qry_chunk_RTs, RT_tolerance
        )
        
    result_bag = product_bag.map(search_block_in_bag)
    logger.info('Initializing precursor search done.')
    if device == 'cpu':
        logger.info('Searching precursor on CPU.')
        results = result_bag.compute(scheduler='threads',num_workers=num_workers)
    else:
        logger.info('Searching precursor on GPU.')
        results = result_bag.compute(scheduler='single-threaded')
    logger.info('Precursor search done.')
    return results