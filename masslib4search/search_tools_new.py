import dask
import dask.bag as db
import dask.dataframe as dd
import dask.array as da
import numba as nb
import numpy as np
import pandas as pd
from rich import print
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

def get_fragments_hits(
    
):
    pass

def search_fragments(
    qry_ions: pd.Series, # shape: (n_ions,)
    ref_fragment_mzs: pd.DataFrame, # shape: (n_ref_fragments, n_adducts), columns: adducts
    ref_fragment_formulas: pd.Series, # shape: (n_ref_fragments,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[NDArray[np.float_]] = None,  # shape: (n_fragments,)
    query_RTs: Optional[NDArray[np.float_]] = None, # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    ref_batch_size: int = 10000,
    qry_chunks: int = 5120,
    ref_chunks: int = 5120,
    I_F_D_matrix_chunks: Tuple[int,int,int] = (10240, 10240, 5),
) -> pd.DataFrame: # columns: db_index, formula, adduct
    
    ref_fragment_mzs_array_block = np.array_split(ref_fragment_mzs.values,ref_batch_size)
    if ref_RTs is not None and query_RTs is not None:
        ref_RT_blocks = np.array_split(ref_RTs,ref_batch_size)
    
    if ref_RTs is not None and query_RTs is not None:
        ref_block_bag = db.from_sequence(zip(ref_fragment_mzs_array_block, ref_RT_blocks))
    else:
        ref_block_bag = db.from_sequence(ref_fragment_mzs_array_block)
        
    

def search_precursors(
    qry_ions: pd.Series, # shape: (n_ions,)
    ref_precursor_mzs: pd.Series, # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[NDArray[np.float_]] = None,  # shape: (n_ref_precursors,)
    query_RTs: Optional[NDArray[np.float_]] = None, # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    batch_size: int = 10,
    qry_chunks: int = 5120,
    ref_chunks: int = 5120,
    Q_R_matrix_chunks: Tuple[int,int] = (10240, 10240),
) -> pd.Series:
    
    qry_index = qry_ions.index
    qry_blocks = np.array_split(qry_ions.values,batch_size)
    if ref_RTs is not None and query_RTs is not None:
        qry_RT_blocks = np.array_split(query_RTs,batch_size)
    
    if ref_RTs is not None and query_RTs is not None:
        qry_bag = db.from_sequence(zip(qry_blocks, qry_RT_blocks), partition_size=1)
    else:
        qry_bag = db.from_sequence(qry_blocks, partition_size=1)
    
    ref_vec = da.from_array(ref_precursor_mzs.values, chunks=ref_chunks)[None,:]
    if ref_RTs is not None and query_RTs is not None:
        ref_RT_vec = da.from_array(ref_RTs, chunks=ref_chunks)[None,:]
    
    def get_QR_matrix(
        qry_block: Union[NDArray[np.float_],Tuple[NDArray[np.float_],NDArray[np.float_]]],
    ) -> da.Array: # shape: (batch_size, n_ref)
        
        if isinstance(qry_block, tuple):
            qry_block, qry_RT_block = qry_block
            qry_RT_block_vec = da.from_array(qry_RT_block, chunks=qry_chunks)[:, None]
        else:
            qry_RT_block = None
            
        qry_block_vec = da.from_array(qry_block, chunks=qry_chunks)[:, None]
        d_matrix = da.abs(qry_block_vec - ref_vec)
        if mz_tolerance_type == 'ppm':
            d_matrix = d_matrix / ref_vec * 1e6
        Q_R_matrix = da.asarray(d_matrix <= mz_tolerance,chunks=Q_R_matrix_chunks)
        
        if qry_RT_block is not None:
            d_matrix_RT = da.abs(qry_RT_block_vec - ref_RT_vec)
            bool_matrix_RT = da.asarray(d_matrix_RT <= RT_tolerance,chunks=Q_R_matrix_chunks)
            Q_R_matrix = Q_R_matrix & bool_matrix_RT
        
        return Q_R_matrix
    
    qry_bag = qry_bag.map(get_QR_matrix)
    
    print('Preparing QR matrix...')
    da_list = dask.compute(qry_bag,scheduler='threads')[0]
    print('Computing QR matrix...')
    np_list = dask.compute(da_list,scheduler='threads')[0]
    Q_R_matrix = np.concatenate(np_list, axis=0)
    
    print('Building result dataframe...')
    ref_index_array = ref_precursor_mzs.index.values
    bool_matrix_db = db.from_sequence(Q_R_matrix, partition_size=1)
    tag_index_db = bool_matrix_db.map(lambda x: ref_index_array[np.where(x)[0]])
    tag_index_db = tag_index_db.map(lambda x: x if len(x) > 0 else 'null')
    tag_db_index = dask.compute(tag_index_db,scheduler='threads')[0]
    return pd.Series(tag_db_index, index=qry_index)

def cosine_similarity_dask(
    query_embeddings: Union[da.Array, NDArray[np.float32]], # shape: (n_query_items, n_dim)
    ref_embeddings: Union[da.Array, NDArray[np.float32]], # shape: (n_ref_items, n_dim)
    query_chunks: int = 5120,
    ref_chunks: int = 5120,
) -> da.Array: # shape: (n_query_items, n_ref_items)
    query_embeddings = da.asarray(query_embeddings, chunks=(query_chunks, -1))
    ref_embeddings = da.asarray(ref_embeddings, chunks=(ref_chunks, -1))
    dot_product = da.dot(query_embeddings, ref_embeddings.T)
    norm_query = da.linalg.norm(query_embeddings, axis=1, keepdims=True)
    norm_ref = da.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    score_matrix = dot_product / (norm_query * norm_ref.T)
    return score_matrix

def cosine_similarity_np(
    query_embeddings: NDArray[np.float32], # shape: (n_dim)
    ref_embeddings: NDArray[np.float32], # shape: (n_ref_items, n_dim)
) -> NDArray[np.float32]: # shape: (n_ref_items)
    query_embeddings = query_embeddings[np.newaxis,:]
    dot_product = np.dot(query_embeddings, ref_embeddings.T)
    norm_query = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    norm_ref = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    score_matrix = dot_product / (norm_query * norm_ref.T)
    return score_matrix[0]

def deocde_index_and_score(
    score_matrix: Union[
        NDArray[np.float32],  # shape: (n_query_items, n_ref_items)
        List[Union[NDArray[np.float32],Literal['null']]],  # len: n_query_items, each element shape: (n_ref_items,)
    ],
    top_k: Optional[int] = None,
) -> Tuple[
        List[Union[NDArray[np.int_],Literal['null']]],  # index of tag ref, len: n_query_items, each element shape: (top_k,)
        List[Union[NDArray[np.float32],Literal['null']]], # score of tag ref, len: n_query_items, each element shape: (top_k,)
]:
    
    score_db = db.from_sequence(score_matrix)
    
    def decoder(
        score_vector:Union[NDArray[np.float32],Literal['null']]
    ) -> Tuple[
        Union[NDArray[np.int_],Literal['null']], 
        Union[NDArray[np.float32],Literal['null']],
    ]:
        if not isinstance(score_vector, str):
            if top_k is None:
                I = np.argsort(score_vector)[::-1]
            else:
                I = np.argsort(score_vector)[::-1][:top_k]
            S = score_vector[I]
        else:
            I,S = 'null','null'
        return I, S
    
    IS_pairs = score_db.map(decoder)
    
    IS_pairs = IS_pairs.compute(scheduler='threads')
    
    I = []
    S = []
    for i_vector, s_vector in IS_pairs:
        I.append(i_vector)
        S.append(s_vector)
    
    return I, S

def search_embeddings(
    qry_embeddings: pd.Series, # Series[1d-NDArray[np.float_]]
    ref_embeddings: pd.Series, # Series[1d-NDArray[np.float_]]
    tag_ref_index: Optional[pd.Series] = None, # Series[1d-NDArray[Hashable] | 'null']
    top_k: Optional[int] = None,
    qry_chunks: int = 5120,
    ref_chunks: int = 5120
) -> pd.DataFrame: # columns: db_index, score
    if tag_ref_index is None:
        print('Calculating score matrix...')
        score_matrix = cosine_similarity_dask(np.stack(qry_embeddings), np.stack(ref_embeddings), qry_chunks, ref_chunks)
        score_matrix = score_matrix.compute(scheduler='threads')
    else:
        print('Initializing embeddings...')
        score_matrix = db.from_sequence(zip(qry_embeddings,tag_ref_index),partition_size=1)
        
        def score_func(item: Tuple[NDArray,Union[NDArray,Literal['null']]]) -> Union[NDArray,Literal['null']]:
            query_vector, ref_index = item
            if len(ref_index) == 0 or isinstance(ref_index, str):
                return 'null'
            ref_vector = np.stack(ref_embeddings[ref_index])
            score_vector = cosine_similarity_np(query_vector, ref_vector)
            return score_vector
        
        score_matrix = score_matrix.map(score_func)
        print('Calculating score matrix by tag index...')
        score_matrix = score_matrix.compute(scheduler='threads')
        
    print('Decoding score matrix...')
    I,S = deocde_index_and_score(score_matrix, top_k)
    if tag_ref_index is not None:
        for i in range(len(I)):
            if not isinstance(I[i], str):
                I[i] = tag_ref_index[i][I[i]]
    return pd.DataFrame({'db_ids': I,'score': S},index=qry_embeddings.index)

@nb.njit
def find_original_index_int(structure_vec:NDArray[np.int64], i_data:int) -> Tuple[int, int]:
    # 使用二分查找定位i_list
    left, right = 0, len(structure_vec)
    while left < right:
        mid = (left + right) // 2
        if structure_vec[mid] <= i_data:
            left = mid + 1
        else:
            right = mid
    i_list = left
    
    # 计算偏移量
    prev_end = structure_vec[i_list-1] if i_list > 0 else 0
    return i_list, i_data - prev_end

@nb.njit
def find_original_index_seq(structure_vec:NDArray[np.int64], i_data:Sequence[int]):
    results = []
    for i in nb.prange(len(i_data)):
        results.append(find_original_index_int(structure_vec, i_data[i]))
    return results

class IrregularArray:
    
    def __init__(self, list_of_arrays: List[NDArray[Any]]):
        self.data_vector = np.concatenate(list_of_arrays)
        self.structure_vector = self._build_structure_vector(list_of_arrays)
    
    def _build_structure_vector(self, list_of_arrays: List[NDArray[Any]]) -> NDArray[np.int64]:
        lengths = np.array([len(arr) for arr in list_of_arrays], dtype=np.int64)
        return np.cumsum(lengths) if len(lengths) > 0 else np.array([], dtype=np.int64)
    
    def find_original_index(self, i_data):
        if isinstance(i_data, int):
            return find_original_index_int(self.structure_vector, i_data)
        else:
            return find_original_index_seq(self.structure_vector, i_data)
        
def dict2list(d: Dict[int, Any], max_length: int, padding_value: Any = None) -> List[Any]:
    re_list = []
    for i in range(max_length):
        if i in d:
            re_list.append(d[i])
        else:
            re_list.append(padding_value)
    return re_list
        
def peak_pattern_search(
    qry_series: pd.Series, 
    ref_series: pd.Series, 
    tag_ref_index: Optional[pd.Series] = None,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    batch_size: int = 10,
    qry_chunks: int = 5120,
    ref_chunks: int = 5120,
    bm_chunks: Tuple[int,int] = (10240, 10240),
) -> pd.DataFrame: # columns: db_index, tag_peaks
    
    qry_blocks = np.array_split(qry_series,batch_size)
    
    qry_bag = db.from_sequence(qry_blocks, partition_size=1)
    
    def mapping_ref(
        qry_block: pd.Series,
    ) -> Tuple[IrregularArray, IrregularArray, pd.Series, pd.Series]:
        
        if tag_ref_index is not None:
            tag_ref_ids = tag_ref_index[qry_block.index]
            tag_ref_ids = np.unique(np.concatenate(tag_ref_ids.to_list()))
            tag_refs = ref_series[tag_ref_ids]
        else:
            tag_refs = ref_series
        
        qry_irr_array = IrregularArray(qry_block.values)
        ref_irr_array = IrregularArray(tag_refs.values)
        
        return qry_irr_array, ref_irr_array, qry_block, tag_refs
    
    def get_bm(
        pair_item: Tuple[IrregularArray, IrregularArray, pd.Series, pd.Series]
    ) -> Tuple[IrregularArray, IrregularArray, pd.Series, pd.Series, da.Array]:
        qry_irr_array, ref_irr_array, qry_block, tag_refs = pair_item
        
        qry_vec = da.from_array(qry_irr_array.data_vector, chunks=qry_chunks).reshape(-1,1)
        ref_vec = da.from_array(ref_irr_array.data_vector, chunks=ref_chunks).reshape(1,-1)
        
        if mz_tolerance_type == 'ppm':
            dm = da.abs(qry_vec - ref_vec) / ref_vec * 1e6
        else:
            dm = da.abs(qry_vec - ref_vec)
        bm:da.Array = dm < mz_tolerance
        bm = da.asarray(bm, chunks=bm_chunks)
        
        return qry_irr_array, ref_irr_array, qry_block, tag_refs, bm
    
    qry_bag = qry_bag.map(mapping_ref)
    qry_bag = qry_bag.map(get_bm)
    
    print('Preparing BM matrix...')
    item_pairs = dask.compute(qry_bag,scheduler='threads')[0]
    print('Computing BM matrix...')
    item_pairs = dask.compute(item_pairs,scheduler='threads')[0]
    
    print('Decoding BM matrix...')
    item_pairs_bag = db.from_sequence(item_pairs, partition_size=1)
    
    def decode(
        pair_item: Tuple[IrregularArray, IrregularArray, pd.Series, pd.Series, NDArray[np.bool_]]
    ) -> Dict[
        Literal['db_ids', 'tag_peaks'],
        Union[List[Hashable], List[List[float]]],
    ]:
        qry_irr_array, ref_irr_array, qry_block, tag_refs, bm = pair_item
        
        q_index,r_index = np.where(bm)

        tag_q_list = qry_irr_array.find_original_index(q_index)
        tag_r_list = ref_irr_array.find_original_index(r_index)
        
        def block_decode(tag_q_list,tag_r_list):
            results:Dict[Hashable,Dict[Hashable,List[float]]] = {}
            for ((q_i,q_j),(r_i,r_j)) in zip(tag_q_list,tag_r_list):
                q_i = qry_block.index[q_i]
                r_i = tag_refs.index[r_i]
                if q_i not in results:
                    results[q_i] = {}
                if r_i not in results[q_i]:
                    results[q_i][r_i] = []
                results[q_i][r_i].append(ref_series[r_i][r_j])
            return results
        
        result_dict = block_decode(tag_q_list,tag_r_list)
        
        for key, value in result_dict.items():
            result_dict[key] = {
                "db_index":list(value.keys()),
                "tag_peaks":list(value.values())
            }
        
        return result_dict
    
    results_bag = item_pairs_bag.map(decode)
    results_list = dask.compute(results_bag,scheduler='threads')[0]
    results_dict = {}
    for result in results_list:
        results_dict.update(result)
    results_list = dict2list(results_dict, len(qry_series), {"db_index":'null',"tag_peaks":'null'})
    results_df = pd.DataFrame(results_list,index=qry_series.index)
    return results_df