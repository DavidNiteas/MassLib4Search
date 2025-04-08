import torch
from .base_search_tools import cosine_similarity_search
import dask.bag as db
import numpy as np
import pandas as pd
import logging
from numpy.typing import NDArray
from typing import Optional, Literal, List, Tuple, Union

logger = logging.getLogger(__name__)

def decode_torch_results(
    score_matrix: Union[np.ndarray, List[np.ndarray]],
    tag_ref_index: Optional[pd.Series] = None,
    top_k: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    
    def decoder(score_vector):
        if isinstance(score_vector, np.ndarray):
            if top_k:
                idx = np.argpartition(score_vector, -top_k)[-top_k:]
                idx = idx[np.argsort(score_vector[idx])[::-1]]
            else:
                idx = np.argsort(score_vector)[::-1]
            return idx, score_vector[idx]
        return 'null', 'null'
    
    return zip(*[decoder(vec) for vec in score_matrix])

def search_embeddings(
    qry_embeddings_queue: List[pd.Series], # List[Series[1d-NDArray[np.float_]]]
    all_ref_embeddings: pd.Series, # Series[1d-NDArray[np.float_]]
    tag_ref_index: Optional[pd.Series] = None,
    top_k: Optional[int] = None,
    device: str = 'cpu',
    num_workers: Optional[int] = None,
    chunk_size: int = 5120,
) -> List[pd.DataFrame]: # columns: db_ids, score
    
    logger.info('Initializing embedding search...')
    qry_queue_bag = db.from_sequence(qry_embeddings_queue)
    
    def search_block_in_bag(qry_embeddings_series: pd.Series) -> pd.DataFrame:
        qry_embeddings = torch.as_tensor(qry_embeddings_series.values, device=device)
        if tag_ref_index is not None:
            ref_embeddings_series = all_ref_embeddings.loc[tag_ref_index]
        else:
            ref_embeddings_series = all_ref_embeddings
        ref_embeddings = torch.as_tensor(ref_embeddings_series.values, device=device)
        if ref_embeddings.numel() != 0 and qry_embeddings.numel() != 0:
            S, I = cosine_similarity_search(qry_embeddings, ref_embeddings, top_k=top_k, chunk_size=chunk_size)
            return pd.DataFrame({'db_ids': I.numpy(), 'score': S.numpy()}, index=qry_embeddings_series.index)
        else:
            return pd.DataFrame({'db_ids': [],'score': []}, index=[])
        
    result_bag = qry_queue_bag.map(search_block_in_bag)
    logger.info('Initializing embedding search done.')
    if device == 'cpu':
        logger.info("Searching embeddings on CPU...")
        results = result_bag.compute(scheduler='threads',num_workers=num_workers)
    else:
        logger.info("Searching embeddings on GPU...")
        results = result_bag.compute(scheduler='single-threaded')
    logger.info("Embedding search done.")
    return results