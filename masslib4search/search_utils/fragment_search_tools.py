import torch
from .base_search_tools import (
    broadcast,get_delta_matrix,ppm_convert,
    get_bool_matrix,adduct_co_occurrence_filter,get_indices
)
import pandas as pd
import numpy as np
import dask.bag as db
import logging
from numpy.typing import NDArray
from typing import Optional, Literal, List, Tuple

logger = logging.getLogger(__name__)

@torch.no_grad()
def get_fragments_hits(
    qry_ions_array: torch.Tensor,          # shape: (n_ions,) float32
    ref_fragment_mzs_array: torch.Tensor,  # shape: (n_ref, n_adducts) float32
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[torch.Tensor] = None, # shape: (n_ref,) float16
    query_RTs: Optional[torch.Tensor] = None, # shape: (n_ions,) float16
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
) -> torch.Tensor:
    
    Q, R = broadcast(qry_ions_array, ref_fragment_mzs_array)
    D = get_delta_matrix(Q, R)
    if mz_tolerance_type == 'ppm':
        D = ppm_convert(D, R)
    B = get_bool_matrix(D, mz_tolerance)
    if ref_RTs is not None and query_RTs is not None:
        RT_Q, RT_R = broadcast(query_RTs, ref_RTs)
        RT_D = get_delta_matrix(RT_Q, RT_R)
        RT_B = get_bool_matrix(RT_D, RT_tolerance)
        B = B & RT_B.view(*RT_B.shape, 1)
        del RT_Q, RT_R, RT_D, RT_B
    if adduct_co_occurrence_threshold > 1:
        B = adduct_co_occurrence_filter(B, adduct_co_occurrence_threshold)
    I = get_indices(B)
    return I

@torch.no_grad()
def search_block_fragments(
    qry_ions_array: torch.Tensor,          # shape: (n_ions,) float32
    ref_fragment_mzs_array: torch.Tensor,  # shape: (n_ref, n_adducts) float32
    qry_row_index: pd.Index,
    ref_row_index: pd.Index,
    ref_col_index: pd.Index,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[torch.Tensor] = None, # shape: (n_ref,) float16
    query_RTs: Optional[torch.Tensor] = None, # shape: (n_ions,) float16
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1,
) -> pd.DataFrame: # columns: db_index, adduct
    I = get_fragments_hits(
        qry_ions_array,ref_fragment_mzs_array,
        mz_tolerance,mz_tolerance_type,
        ref_RTs,query_RTs,RT_tolerance,
        adduct_co_occurrence_threshold,
    ).to("cpu")
    hitted_qry = qry_row_index[I[:, 0]]
    hitted_ref = ref_row_index[I[:, 1]]
    hitted_adduct = ref_col_index[I[:, 2]]
    return pd.DataFrame({'db_ids': hitted_ref, 'adduct': hitted_adduct}, index=hitted_qry)

@torch.no_grad()
def search_fragments(
    qry_ions_array_queue: List[pd.Series], # shape: (n_ions,)
    ref_fragment_mzs_queue: List[pd.DataFrame], # shape: (n_ref_fragments, n_adducts), columns: adducts
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[NDArray[np.float_]] = None,  # shape: (n_fragments,)
    query_RTs: Optional[NDArray[np.float_]] = None, # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    device: str = 'cpu',
    num_workers: Optional[int] = None,
) -> List[pd.DataFrame]: # columns: db_index, formula, adduct
    
    logger.info('Initializing fragments search...')
    qry_queue_bag = db.from_sequence(qry_ions_array_queue)
    ref_queue_bag = db.from_sequence(ref_fragment_mzs_queue)
    queue_bag = qry_queue_bag.product(ref_queue_bag)
    
    def search_block_in_bag(chunk: Tuple[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        qry_ions_series, ref_fragments_df = chunk
        qry_row_index = qry_ions_series.index
        ref_row_index = ref_fragments_df.index
        ref_col_index = ref_fragments_df.columns
        qry_ions_array = torch.as_tensor(qry_ions_series.values, dtype=torch.float32, device=device)
        ref_fragment_mzs_array = torch.as_tensor(ref_fragments_df.values, dtype=torch.float32, device=device)
        ref_chunk_RTs = torch.as_tensor(ref_RTs[ref_row_index], dtype=torch.float16, device=device) if ref_RTs is not None else None
        qry_chunk_RTs = torch.as_tensor(query_RTs[qry_row_index], dtype=torch.float16, device=device) if query_RTs is not None else None
        return search_block_fragments(
            qry_ions_array, ref_fragment_mzs_array,
            qry_row_index, ref_row_index, ref_col_index,
            mz_tolerance, mz_tolerance_type,
            ref_chunk_RTs, qry_chunk_RTs, RT_tolerance,
            adduct_co_occurrence_threshold,
        )
        
    result_bag = queue_bag.map(search_block_in_bag)
    logger.info('Initializing fragments search done.')
    if device == 'cpu':
        logger.info('Searching fragments on CPU...')
        results = result_bag.compute(scheduler='threads',num_workers=num_workers)
    else:
        logger.info('Searching fragments on GPU...')
        results = result_bag.compute(scheduler='single-threaded')
    logger.info('Fragments search done.')
    return results