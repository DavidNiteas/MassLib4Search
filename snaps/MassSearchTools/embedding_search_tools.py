import torch
from .utils.cosine_search import cosine_similarity_search
import dask.bag as db
import numpy as np
import pandas as pd
import logging
from numpy.typing import NDArray
from typing import Optional, Literal, List, Tuple, Union

logger = logging.getLogger(__name__)

def search_embeddings(
    qry_embeddings_queue: List[pd.DataFrame], # List[DataFrame[index: qry_ids, columns: range(n_features)]]
    all_ref_embeddings: pd.DataFrame, # DataFrame[index: ref_ids, columns: range(n_features)]
    tag_ref_index_queue: Optional[List[Optional[pd.Series]]] = None,
    top_k: Optional[int] = None,
    device: str = 'cpu',
    num_workers: Optional[int] = None,
    chunk_size: int = 5120,
) -> List[pd.DataFrame]: # columns: qry_ids, ref_ids, score
    
    logger.info('Initializing embedding search...')
    if tag_ref_index_queue is None:
        tag_ref_index_queue = [None] * len(qry_embeddings_queue)
    qry_queue_bag = db.from_sequence(zip(qry_embeddings_queue,tag_ref_index_queue))
    
    def search_block_in_bag(qry_embeddings_series: pd.Series, tag_ref_index: Optional[pd.Series]) -> pd.DataFrame:
        qry_embeddings = torch.as_tensor(qry_embeddings_series.values, device=device)
        if tag_ref_index is not None:
            runtime_ref_embeddings = all_ref_embeddings.loc[tag_ref_index]
        else:
            runtime_ref_embeddings = all_ref_embeddings
        ref_embeddings = torch.as_tensor(runtime_ref_embeddings.values, device=device)
        if ref_embeddings.numel() != 0 and qry_embeddings.numel() != 0:
            S, I = cosine_similarity_search(qry_embeddings, ref_embeddings, top_k=top_k, chunk_size=chunk_size)
            qry_ids = np.repeat(qry_embeddings_series.index.values, S.shape[1])
            ref_ids = runtime_ref_embeddings.index.values[I.flatten()]
            score = S.flatten().numpy()
            return pd.DataFrame({'qry_ids':qry_ids, 'ref_ids':ref_ids, 'score':score})
        else:
            return pd.DataFrame(columns=['qry_ids', 'ref_ids', 'score'])
        
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