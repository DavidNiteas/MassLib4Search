import dask
import dask.bag as db
import dask.dataframe as dd
import dask.array as da
from dask import delayed
from dask.delayed import Delayed
import numpy as np
import pandas as pd
# from mzinferrer.mz_infer_tools import Fragment
import pickle
import rich.progress
from rich.progress import track
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

# dev import
import sys
sys.path.append('.')
from MZInferrer.mzinferrer.mz_infer_tools import Fragment,Adduct

def to_pickle_bytes(obj) -> bytes:
    return pickle.dumps(obj)

def from_pickle_bytes(b: bytes) -> Any:
    return pickle.loads(b)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pickle(
    path: str,
    desc: Optional[str] = None,
    **kwargs,
):
    if desc is None:
        desc = f"Loading data from {path}"
    with rich.progress.open(path, 'rb', description=desc, **kwargs) as f:
        return pickle.load(f)

def dict2list(d: Dict[int, Any], max_length: int, padding_value: Any = None) -> List[Any]:
    re_list = []
    for i in range(max_length):
        if i in d:
            re_list.append(d[i])
        else:
            re_list.append(padding_value)
    return re_list

def smiles2formula(smiles: str) -> Optional[str]:
    try:
        formula:str = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))
        if "+" in formula:
            return formula.split("+")[0]
        if "-" in formula:
            return formula.split("-")[0]
        return formula
    except:
        return None
    
def predict_adduct_mz(adduct: Adduct, mass: Union[float, None]) -> Union[float, None]:
    if mass is None:
        return None
    return adduct.predict_mz(mass)

def search_fragments_to_matrix(
    ref_fragments: Union[da.Array, np.ndarray],  # shape: (n_fragments, n_adducts)
    query_ions: Union[da.Array, np.ndarray],  # shape: (n_ions,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[Union[da.Array, np.ndarray]] = None,  # shape: (n_fragments,)
    query_RTs: Optional[Union[da.Array, np.ndarray]] = None,   # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
) -> da.Array:  # shape: (n_ions, n_fragments, n_adducts)
    '''
    该函数生成一个布尔矩阵，用于指示给定的离子是否可以与给定的加成物形成片段。
    - 矩阵具有三个维度：(n_ions, n_fragments, n_adducts)。
    - 矩阵中 (i, j, k) 位置的值为 True 表示离子 i 可以与加成物 k 形成片段 j，否则为 False。
    - 如果 fragment_RTs 和 ion_RTs 不为 None，函数还将根据 RT 条件进行过滤。
    - 如果某个公式的加成物数量少于 adduct_co_occurrence_threshold，则该碎片将从结果中移除。
    - 该函数是一个惰性操作，返回一个 dask 数组。如果您想要计算结果，请使用 `dask.compute()` 方法。

    This function generates a boolean matrix indicating whether a given ion can form a fragment with a given adduct.
    - The matrix has three dimensions: (n_ions, n_fragments, n_adducts).
    - The value at (i, j, k) is True if ion i can form fragment j with adduct k, and False otherwise.
    - The function can handle both m/z and RT conditions.
    - If fragment_RTs and ion_RTs are not None, the function will also filter by RT condition.
    - If a formula has fewer than adduct_co_occurrence_threshold adducts, it will be removed from the result.
    - This function is a lazy operation and returns a dask array. If you want to compute the result, use the `dask.compute()` method.
    '''
    
    # convert to dask arrays
    ref_fragments = da.asarray(ref_fragments)
    query_ions = da.asarray(query_ions)
    
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
        ref_RTs = da.asarray(ref_RTs)
        query_RTs = da.asarray(query_RTs)
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

def decode_matrix_to_fragments(
    formulas: pd.Series,
    ref_fragment_table: pd.DataFrame, # shape: (n_fragments, n_adducts), columns: adducts
    query_ions: np.ndarray, # shape: (n_ions,)
    indices: np.ndarray, # shape: (ion_index, formula_index, adduct_index)
) -> List[Dict[str, str]]: # List[Dict[formula, adduct]]
    results: Dict[int, Dict[str, str]] = {}
    for ion_index, formula_index, adduct_index in indices:
        if ion_index not in results:
            results[ion_index] = {}
        formula = formulas[formula_index]
        results[ion_index][formula] = ref_fragment_table.columns[adduct_index]
    results = dict2list(results, len(query_ions), {})
    return results

def search_fragments(
    formulas: pd.Series,
    ref_fragment_table: Union[pd.DataFrame,dd.DataFrame], # shape: (n_fragments, n_adducts), columns: adducts
    query_ions: Union[da.Array, np.ndarray], # shape: (n_ions,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    ref_RTs: Optional[Union[da.Array, np.ndarray]] = None,  # shape: (n_fragments,)
    query_RTs: Optional[Union[da.Array, np.ndarray]] = None, # shape: (n_ions,)
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    return_matrix: bool = False, # if True, return a boolean matrix indicating whether a given ion can form a fragment with a given adduct.
) -> Union[
    List[Dict[str, str]], # List[Dict[formula, adduct]]
    Tuple[List[Dict[str, str]], np.ndarray],
]:
    '''
    该函数生成一个字典列表，用于表征问询离子与参考碎片的配对情况。
    - 字典列表的元素为字典，键为命中的分子式，值为加合类型。
    - 如果 return_matrix 为 True，函数还将返回一个布尔矩阵，是search_fragments_to_matrix函数的输出。
    - 如果你在使用dask，请注意，该函数在内部调用了dask.compute()方法，这可能会影响你的其他计算图，特别是你输入的数组是dask数组。
    
    This function generates a list of dictionaries, representing the pairing of query ions with reference fragments.
    - The elements of the list are dictionaries, where the keys are the formulas of the hit formulas, and the values are the adduct types.
    - If return_matrix is True, the function will also return a boolean matrix, which is the output of the search_fragments_to_matrix function.
    - Note that this function internally calls the dask.compute() method, which may affect your other computations, especially if your input arrays are dask arrays.
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
    
    # get indices
    indices = da.argwhere(bool_matrix)
    
    # run computation graph
    bool_matrix, indices, ref_fragment_table, query_ions = dask.compute(bool_matrix, indices, ref_fragment_table, query_ions)
    
    # decode results
    results = decode_matrix_to_fragments(formulas, ref_fragment_table, query_ions, indices)
    if return_matrix:
        return results, bool_matrix
    return results

# def search_embeddings_faiss(
#     faiss_index: faiss.Index,
#     embeddings: np.ndarray, # shape: (n_items, n_dim)
#     top_k: int = 1000,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     S, I = faiss_index.search(embeddings, k=top_k)
#     return I[0], S[0]

def cosine_similarity(
    query_embeddings: Union[da.Array, np.ndarray], # shape: (n_query_items, n_dim)
    ref_embeddings: Union[da.Array, np.ndarray], # shape: (n_ref_items, n_dim)
) -> da.Array: # shape: (n_query_items, n_ref_items)
    query_embeddings = da.asarray(query_embeddings)
    ref_embeddings = da.asarray(ref_embeddings)
    dot_product = da.dot(query_embeddings, ref_embeddings.T)
    norm_query = da.linalg.norm(query_embeddings, axis=1, keepdims=True)
    norm_ref = da.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    score_matrix = dot_product / (norm_query * norm_ref.T)
    return score_matrix

def infer_I_S_from_score_matrix(
    score_matrix: Union[da.Array, np.ndarray], # shape: (n_query_items, n_ref_items)
    top_k: int,
    ) -> Tuple[da.Array, da.Array]: # shape: (n_query_items, top_k), (n_query_items, top_k)
    score_matrix = da.asarray(score_matrix)
    I = da.argtopk(-score_matrix, top_k, axis=-1)
    S = []
    for i in range(len(score_matrix)):
        S.append(da.take(score_matrix[i], I[i]))
    S = da.stack(S,axis=0)
    return I, S

def search_embeddings(
    query_embeddings: Union[da.Array, np.ndarray], # shape: (n_query_items, n_dim)
    ref_embeddings: Union[da.Array, np.ndarray], # shape: (n_ref_items, n_dim)
    mask: Optional[Union[da.Array, np.ndarray]] = None, # shape: (n_query_items,n_ref_items)
    top_k: Optional[int] = None,
    return_no_mask: bool = False,
) -> Union[
    Tuple[da.Array, da.Array],
    Tuple[da.Array, da.Array, da.Array, da.Array], # if return_no_mask is True
]:
    if top_k is None:
        top_k = len(ref_embeddings)
    score_matrix = cosine_similarity(query_embeddings, ref_embeddings)
    if mask is not None:
        mask = da.asarray(mask)
        masked_score_matrix = score_matrix * mask
    else:
        masked_score_matrix = score_matrix
    I, S = infer_I_S_from_score_matrix(masked_score_matrix, top_k)
    if return_no_mask:
        I_no_mask, S_no_mask = infer_I_S_from_score_matrix(score_matrix, top_k)
        return I, S, I_no_mask, S_no_mask
    else:
        return I, S

def infer_precursors_table(
    PI: Optional[List[float]] = None,
    RT: Optional[List[float]] = None,
) -> pd.DataFrame:
    df = pd.DataFrame()
    if PI is not None:
        df['PI'] = PI
    if RT is not None:
        df['RT'] = RT
    return df

def infer_peaks_table(
    mzs: Optional[List[List[float]]] = None,
    intensities: Optional[List[List[float]]] = None,
) -> pd.DataFrame:
    df = pd.DataFrame()
    if mzs is not None:
        df['mzs'] = mzs
    if intensities is not None:
        df['intensities'] = intensities
    return df
    
def search_precursors_to_matrix(
    query_precursor_mzs: Union[da.Array, np.ndarray], # shape: (n_query_precursors,)
    ref_precursor_mzs: Union[da.Array, np.ndarray], # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    query_precursor_RTs: Optional[Union[da.Array, np.ndarray]] = None,
    ref_precursor_RTs: Optional[Union[da.Array, np.ndarray]] = None,
    RT_tolerance: float = 0.1,
) -> da.Array: # shape: (n_query_precursors, n_ref_precursors)
    
    # convert to dask arrays
    query_precursor_mzs = da.asarray(query_precursor_mzs)
    ref_precursor_mzs = da.asarray(ref_precursor_mzs)
    
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
        query_precursor_RTs = da.asarray(query_precursor_RTs)
        ref_precursor_RTs = da.asarray(ref_precursor_RTs)
        # calculate RT difference matrix
        d_matrix_RT = da.abs(query_precursor_RTs[:, None] - ref_precursor_RTs[None, :])
        bool_matrix_RT = d_matrix_RT <= RT_tolerance
        # combine RT condition with mz condition
        Q_R_matrix = Q_R_matrix & bool_matrix_RT
    
    return Q_R_matrix

def decode_precursors_matrix_to_index(
    query_lens: int,
    indices: np.ndarray, # shape: (query_index, ref_index)
) -> List[List[int]]: # List[List[ref_precursor_index]]
    results: Dict[int, List[int]] = {}
    for query_index, ref_index in indices:
        if query_index not in results:
            results[query_index] = []
        results[query_index].append(ref_index)
    results = dict2list(results, query_lens, [])
    return results

def search_precursors(
    query_precursor_mzs: Union[da.Array, np.ndarray], # shape: (n_query_precursors,)
    ref_precursor_mzs: Union[da.Array, np.ndarray], # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    query_precursor_RTs: Optional[Union[da.Array, np.ndarray]] = None,
    ref_precursor_RTs: Optional[Union[da.Array, np.ndarray]] = None,
    RT_tolerance: float = 0.1,
    return_matrix: bool = False,
) -> Union[
    List[List[int]], # List[List[ref_precursor_index]]
    Tuple[List[List[int]], np.ndarray],
]:
    bool_matrix = search_precursors_to_matrix(
        query_precursor_mzs, ref_precursor_mzs, mz_tolerance, mz_tolerance_type, 
        query_precursor_RTs, ref_precursor_RTs, RT_tolerance,
    )
    indices = da.argwhere(bool_matrix)
    if return_matrix:
        (indices,bool_matrix) = dask.compute(indices,bool_matrix)
        results =  decode_precursors_matrix_to_index(indices)
        return results, bool_matrix
    else:
        (indices,) = dask.compute(indices)
        results = decode_precursors_matrix_to_index(indices)
        return results