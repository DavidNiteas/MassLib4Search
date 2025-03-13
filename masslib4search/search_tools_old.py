import dask
import dask.bag as db
import dask.dataframe as dd
import dask.array as da
import numba as nb
import numpy as np
import pandas as pd
# from mzinferrer.mz_infer_tools import Fragment
from rich import print
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

def dict2list(d: Dict[int, Any], max_length: int, padding_value: Any = None) -> List[Any]:
    re_list = []
    for i in range(max_length):
        if i in d:
            re_list.append(d[i])
        else:
            re_list.append(padding_value)
    return re_list

def search_fragments_to_matrix(
    ref_fragments: Union[da.Array, NDArray[np.float_]],  # shape: (n_fragments, n_adducts)
    query_ions: Union[da.Array, NDArray[np.float_]],  # shape: (n_ions,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,  # shape: (n_fragments,)
    query_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,   # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
) -> da.Array:  # shape: (n_ions, n_fragments, n_adducts)
    '''
    根据给定的质荷比(mz)和保留时间(RT)的容差值，在参考碎片表中搜索与查询离子匹配的碎片。

    参数:
    - ref_fragments: 
        参考碎片表的数组，可以是dask数组或numpy数组，形状为(n_fragments, n_adducts)。
    - query_ions: 
        查询离子的数组，可以是dask数组或numpy数组，形状为(n_ions,)。
    - mz_tolerance: 
        质荷比容差值，默认为3。
    - mz_tolerance_type: 
        质荷比容差类型，'ppm'或'Da'，默认为'ppm'。
    - ref_RTs: 
        参考碎片表中的保留时间数组，可以是dask数组或numpy数组，形状为(n_fragments,)，可选。
    - query_RTs: 
        查询离子的保留时间数组，可以是dask数组或numpy数组，形状为(n_ions,)，可选。
    - RT_tolerance: 
        保留时间容差值，默认为0.1分钟。
    - adduct_co_occurrence_threshold: 
        如果某个公式的共现加成型少于这个阈值，则从结果中移除该公式，默认为1。

    返回值:
    - 一个布尔dask数组，形状为(n_ions, n_fragments, n_adducts)，指示每个查询离子与每个参考碎片表中的碎片和加成型的匹配情况。
    '''
    
    # convert to dask arrays
    ref_fragments = da.asarray(ref_fragments,chunks=(5120, 1))
    query_ions = da.asarray(query_ions,chunks=(5120,))
    
    # calculate mass difference matrix
    d_matrix = da.abs(query_ions[:, None, None] - ref_fragments[None, :, :])
    
    # convert ppm tolerance to Da tolerance
    if mz_tolerance_type == 'ppm':
        d_matrix = d_matrix / ref_fragments[None, :, :] * 1e6
    
    # generate boolean matrix
    I_F_D_matrix = d_matrix <= mz_tolerance
    
    # handle RT condition
    if ref_RTs is not None and query_RTs is not None:
        # convert to dask arrays
        ref_RTs = da.asarray(ref_RTs,chunks=(5120,))
        query_RTs = da.asarray(query_RTs,chunks=(5120,))
        # calculate RT difference matrix
        d_matrix_RT = da.abs(query_RTs[:, None] - ref_RTs[None, :])
        bool_matrix_RT = d_matrix_RT <= RT_tolerance
        # combine RT condition with mz condition
        I_F_D_matrix = I_F_D_matrix & bool_matrix_RT[:, :, None]
    
    # filter by adduct co-occurrence
    F_D_matrix = da.sum(I_F_D_matrix, axis=0, keepdims=False)
    F_matrix = da.sum(F_D_matrix, axis=1, keepdims=False)
    F_matrix = F_matrix >= adduct_co_occurrence_threshold
    I_F_D_matrix = I_F_D_matrix & F_matrix[None, :, None]
    
    return I_F_D_matrix

def decode_matrix_to_db_index(
    bool_matrix: NDArray[np.bool_], # shape: (n_ions, n_fragments, n_adducts)
    select_list: Optional[List[Optional[NDArray[np.int_]]]] = None, # List[1d-array]
    return_bag: bool = True,
) -> Union[List[NDArray[np.int_]], db.Bag]:
    if select_list is None:
        bool_matrix_db = db.from_sequence(bool_matrix, partition_size=1)
        tag_index_db = bool_matrix_db.map(lambda x: np.where(x))
        tag_index_db = tag_index_db.map(lambda x: x if len(x[0]) > 0 else None)
    else:
        tag_index_db = db.from_sequence(zip(bool_matrix, select_list), partition_size=1)

        def get_index(x: Tuple[NDArray[np.bool_], Optional[NDArray[np.int_]]]) -> Optional[NDArray[np.int_]]:
            item_bool_matrix, index_vector = x
            if index_vector is None:
                return None
            if len(index_vector) == 0:
                return None
            tag_index = []
            for adduct_vector, formula_index in zip(item_bool_matrix[index_vector,:], index_vector):
                adduct_index = np.where(adduct_vector)[0].item()
                tag_index.append([formula_index, adduct_index])
            return np.array(tag_index)
        
        tag_index_db = tag_index_db.map(get_index)
    if return_bag:
        return tag_index_db
    else:
        tag_index_db = tag_index_db.compute(scheduler='threads')
        return tag_index_db    

def decode_matrix_to_fragments(
    formulas: pd.Series,
    adducts: pd.Series,
    bool_matrix: NDArray[np.bool_], # shape: (n_ions, n_fragments, n_adducts)
    select_list: Optional[List[Optional[NDArray[np.int_]]]] = None, # List[1d-array]
) -> Dict[
    Literal['formula', 'adduct', 'db_index'],
    Union[
        List[NDArray[np.str_]], # formula and adduct, list len: query item num, array shape: (tag_formula_num,)
        List[NDArray[np.int_]], # db_index, list len: query item num, array shape: (tag_formula_num,)
    ],
]:
    tag_index_db = decode_matrix_to_db_index(bool_matrix, select_list, return_bag=True)
    tag_formula = tag_index_db.map(lambda x: formulas[x[0]].values if x is not None else None)
    tag_adduct = tag_index_db.map(lambda x: adducts[x[1]].values if x is not None else None)
    tag_formula,tag_adduct,tag_db_index = dask.compute(tag_formula,tag_adduct,tag_index_db,scheduler='threads')
    return {'formula': tag_formula, 'adduct': tag_adduct, 'db_index': tag_db_index}

def search_fragments(
    formulas: pd.Series,
    ref_fragment_table: Union[pd.DataFrame,dd.DataFrame], # shape: (n_fragments, n_adducts), columns: adducts
    query_ions: Union[da.Array, NDArray[np.float_]], # shape: (n_ions,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,  # shape: (n_fragments,)
    query_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None, # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    return_matrix: bool = False, # if True, return a boolean matrix indicating whether a given ion can form a fragment with a given adduct.
) -> Union[
        Dict[
            Literal['formula', 'adduct', 'db_index'],
            Union[
                List[NDArray[np.str_]], # formula and adduct, list len: query item num, array shape: (tag_formula_num,)
                List[NDArray[np.int_]], # db_index, list len: query item num, array shape: (tag_formula_num,)
            ],
        ],
        Tuple[
            Dict[
                Literal['formula', 'adduct', 'db_index'],
                Union[
                    List[NDArray[np.str_]], # formula and adduct, list len: query item num, array shape: (tag_formula_num,)
                    List[NDArray[np.int_]], # db_index, list len: query item num, array shape: (tag_formula_num,)
                ],
            ],
            NDArray[np.bool_] # bool_matrix, shape: (n_ions, n_fragments, n_adducts)
        ], # if return_matrix is True, return a boolean matrix indicating whether a given ion can form a fragment with a given adduct.
]:
    '''
        该函数用于在给定的参考碎片表中搜索与查询离子匹配的碎片。匹配基于质荷比(mz)和保留时间(RT)的容差值。函数可以返回匹配的公式、加成型和数据库索引，
        或者如果设置了return_matrix为True，则同时返回布尔矩阵，指示每个查询离子与每个参考碎片表中的碎片和加成型的匹配情况。

        ### 参数:
        - `formulas`: 包含待查询公式的pandas Series。
        - `ref_fragment_table`: 参考碎片表，可以是pandas DataFrame或dask DataFrame。
        - `query_ions`: 待查询离子的数组，可以是dask数组或numpy数组。
        - `mz_tolerance`: 质荷比容差值，默认为3。
        - `mz_tolerance_type`: 质荷比容差类型，'ppm'或'Da'，默认为'ppm'。
        - `ref_RTs`: 参考碎片表中的保留时间数组，可以是dask数组或numpy数组，可选。
        - `query_RTs`: 查询离子的保留时间数组，可以是dask数组或numpy数组，可选。
        - `RT_tolerance`: 保留时间容差值，默认为0.1分钟。
        - `adduct_co_occurrence_threshold`: 如果某个公式的共现加成型少于这个阈值，则从结果中移除该公式，默认为1。
        - `return_matrix`: 如果为True，则返回布尔矩阵指示匹配情况，默认为False。

        ### 返回值:
        - 如果`return_matrix`为False，则返回一个字典，包含匹配的公式、加成型和数据库索引。
        - 如果`return_matrix`为True，则返回一个包含匹配结果字典和布尔矩阵的元组。
    '''
    # convert to dask arrays
    if isinstance(ref_fragment_table, pd.DataFrame):
        ref_fragments = ref_fragment_table.values
    else:
        ref_fragments = ref_fragment_table.to_dask_array(lengths=True)
    
    # get boolean matrix
    bool_matrix = search_fragments_to_matrix(
        ref_fragments, query_ions, mz_tolerance, mz_tolerance_type, 
        ref_RTs, query_RTs, RT_tolerance, 
        adduct_co_occurrence_threshold,
    )
    
    # run computation graph
    print('Calculating fragments hit matrix...')
    bool_matrix, ref_fragment_table, query_ions = dask.compute(bool_matrix, ref_fragment_table, query_ions)
    
    # decode results
    print('Decoding fragments from hit matrix...')
    adduct = pd.Series(ref_fragment_table.columns.values)
    results = decode_matrix_to_fragments(formulas, adduct, bool_matrix)
    if return_matrix:
        return results, bool_matrix
    return results

def cosine_similarity_dask(
    query_embeddings: Union[da.Array, NDArray[np.float32]], # shape: (n_query_items, n_dim)
    ref_embeddings: Union[da.Array, NDArray[np.float32]], # shape: (n_ref_items, n_dim)
) -> da.Array: # shape: (n_query_items, n_ref_items)
    query_embeddings = da.asarray(query_embeddings, chunks=(5120, -1))
    ref_embeddings = da.asarray(ref_embeddings, chunks=(5120, -1))
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
        List[NDArray[np.float32]],  # len: n_query_items, each element shape: (n_ref_items,)
    ],
    top_k: Optional[int] = None,
) -> Tuple[
        List[Optional[NDArray[np.int_]]],  # index of tag ref, len: n_query_items, each element shape: (top_k,)
        List[Optional[NDArray[np.float32]]], # score of tag ref, len: n_query_items, each element shape: (top_k,)
]:
    
    score_db = db.from_sequence(score_matrix)
    
    def decoder(score_vector:Optional[NDArray[np.float32]]) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        if score_vector is not None:
            if top_k is None:
                I = np.argsort(score_vector)[::-1]
            else:
                I = np.argsort(score_vector)[::-1][:top_k]
            S = score_vector[I]
        else:
            I,S = None,None
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
    query_embeddings: Union[da.Array, NDArray[np.float32]], # shape: (n_query_items, n_dim)
    ref_embeddings: Union[da.Array, NDArray[np.float32]], # shape: (n_ref_items, n_dim)
    tag_index: Optional[List[NDArray[np.int_]]] = None,
    top_k: Optional[int] = None,
) -> Tuple[
        List[Optional[NDArray[np.int_]]],  # index of tag ref, len: n_query_items, each element shape: (top_k,)
        List[Optional[NDArray[np.float32]]], # score of tag ref, len: n_query_items, each element shape: (top_k,)
]:
    if tag_index is None:
        print('Calculating score matrix...')
        score_matrix = cosine_similarity_dask(query_embeddings, ref_embeddings)
        score_matrix = score_matrix.compute(scheduler='threads')
    else:
        print('Initializing embeddings...')
        query_embeddings, ref_embeddings = dask.compute(query_embeddings, ref_embeddings)
        score_matrix = db.from_sequence(zip(query_embeddings,tag_index),partition_size=1)
        
        def score_func(item: Tuple[NDArray,NDArray]) -> Optional[NDArray]:
            query_vector, ref_index = item
            if len(ref_index) == 0:
                return None
            ref_vector = ref_embeddings[ref_index]
            score_vector = cosine_similarity_np(query_vector, ref_vector)
            return score_vector
        
        score_matrix = score_matrix.map(score_func)
        print('Calculating score matrix by tag index...')
        score_matrix = score_matrix.compute(scheduler='threads')
    
    print('Decoding score matrix...')
    I,S = deocde_index_and_score(score_matrix, top_k)
    if tag_index is not None:
        for i in range(len(I)):
            if I[i] is not None:
                I[i] = tag_index[i][I[i]]
    return I,S
    
def search_precursors_to_matrix(
    query_precursor_mzs: Union[da.Array, NDArray[np.float_]], # shape: (n_query_precursors,)
    ref_precursor_mzs: Union[da.Array, NDArray[np.float_]], # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    query_precursor_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,
    ref_precursor_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,
    RT_tolerance: float = 0.1,
) -> da.Array: # shape: (n_query_precursors, n_ref_precursors)
    
    # convert to dask arrays
    query_precursor_mzs = da.asarray(query_precursor_mzs,chunks=(5120,))
    ref_precursor_mzs = da.asarray(ref_precursor_mzs,chunks=(5120,))
    
    # calculate mass difference matrix
    d_matrix = da.abs(query_precursor_mzs[:, None] - ref_precursor_mzs[None,:])
    
    # convert ppm tolerance to Da tolerance
    if mz_tolerance_type == 'ppm':
        d_matrix = d_matrix / query_precursor_mzs[None,:] * 1e6
    
    # generate boolean matrix
    Q_R_matrix = d_matrix <= mz_tolerance
    
    # handle RT condition
    if query_precursor_RTs is not None and ref_precursor_RTs is not None:
        # convert to dask arrays
        query_precursor_RTs = da.asarray(query_precursor_RTs,chunks=(5120,))
        ref_precursor_RTs = da.asarray(ref_precursor_RTs,chunks=(5120,))
        # calculate RT difference matrix
        d_matrix_RT = da.abs(query_precursor_RTs[:, None] - ref_precursor_RTs[None, :])
        bool_matrix_RT = d_matrix_RT <= RT_tolerance
        # combine RT condition with mz condition
        Q_R_matrix = Q_R_matrix & bool_matrix_RT
    
    return Q_R_matrix

def decode_precursors_matrix_to_index(
    query_lens: int,
    indices: Tuple[NDArray[np.int_], NDArray[np.int_]],
) -> List[List[int]]: # List[List[ref_precursor_index]]
    results: Dict[int, List[int]] = {}
    for query_index, ref_index in zip(indices):
        if query_index not in results:
            results[query_index] = []
        results[query_index].append(ref_index)
    results = dict2list(results, query_lens, [])
    return results

def search_precursors(
    query_precursor_mzs: Union[da.Array, NDArray[np.float_]], # shape: (n_query_precursors,)
    ref_precursor_mzs: Union[da.Array, NDArray[np.float_]], # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    query_precursor_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,
    ref_precursor_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,
    RT_tolerance: float = 0.1,
    return_matrix: bool = False,
) -> Union[
    List[List[int]], # List[List[ref_precursor_index]]
    Tuple[List[List[int]], NDArray[np.bool_]],
]:
    bool_matrix = search_precursors_to_matrix(
        query_precursor_mzs, ref_precursor_mzs, mz_tolerance, mz_tolerance_type, 
        query_precursor_RTs, ref_precursor_RTs, RT_tolerance,
    )
    bool_matrix = dask.compute(bool_matrix)
    indices = np.where(bool_matrix)
    results =  decode_precursors_matrix_to_index(len(query_precursor_mzs),indices)
    if return_matrix:
        return results, bool_matrix
    else:
        return results
    
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
        
def peak_pattern_search(
    qry_series: pd.Series, 
    ref_series: pd.Series, 
    tag_ref_index: Optional[pd.Series] = None,
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    batch_size: int = 10,
) -> List[Dict[Hashable, List[float]]]:
    
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
        
        qry_vec = da.from_array(qry_irr_array.data_vector, chunks=5120).reshape(-1,1)
        ref_vec = da.from_array(ref_irr_array.data_vector, chunks=5120).reshape(1,-1)
        
        if mz_tolerance_type == 'ppm':
            dm = da.abs(qry_vec - ref_vec) / ref_vec * 1e6
        else:
            dm = da.abs(qry_vec - ref_vec)
        bm:da.Array = dm < mz_tolerance
        bm = da.asarray(bm, chunks=(10240,10240))
        
        return qry_irr_array, ref_irr_array, qry_block, tag_refs, bm
    
    qry_bag = qry_bag.map(mapping_ref)
    qry_bag = qry_bag.map(get_bm)
    
    item_pairs = dask.compute(qry_bag,scheduler='threads')[0]
    item_pairs = dask.compute(item_pairs,scheduler='threads')[0]
    
    item_pairs_bag = db.from_sequence(item_pairs, partition_size=1)
    
    def decode(
        pair_item: Tuple[IrregularArray, IrregularArray, pd.Series, pd.Series, NDArray[np.bool_]]
    ) -> Dict[Hashable, List[float]]:
        qry_irr_array, ref_irr_array, qry_block, tag_refs, bm = pair_item
        
        q_index,r_index = np.where(bm)

        tag_q_list = qry_irr_array.find_original_index(q_index)
        tag_r_list = ref_irr_array.find_original_index(r_index)
        
        def block_decode(tag_q_list,tag_r_list):
            results:Dict[Hashable,Dict[Hashable,List[Tuple[Hashable,Hashable]]]] = {}
            for ((q_i,q_j),(r_i,r_j)) in zip(tag_q_list,tag_r_list):
                q_i = qry_block.index[q_i]
                r_i = tag_refs.index[r_i]
                if q_i not in results:
                    results[q_i] = {}
                if r_i not in results[q_i]:
                    results[q_i][r_i] = []
                results[q_i][r_i].append((q_j, r_j))
            return results
        
        result_dict = block_decode(tag_q_list,tag_r_list)
        
        return result_dict
    
    results_bag = item_pairs_bag.map(decode)
    results_list = dask.compute(results_bag,scheduler='threads')[0]
    results_dict = {}
    for result in results_list:
        results_dict.update(result)
    results_list = dict2list(results_dict, len(qry_series), [])
    return results_list