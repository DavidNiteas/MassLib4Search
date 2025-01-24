import numpy as np
import modin.pandas as pd
from mzinferrer.mz_infer_tools import Fragment
import pickle
import rich.progress
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
# import faiss
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

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

def smiles2formula(smiles: List[str]) -> List[str]:
    smiles: pd.Series = pd.Series(smiles)
    formulas = smiles.apply(lambda x: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(x)))
    return formulas.tolist

def infer_fragments_table(
    formula: List[str],
    adducts: List[str],
    RT: Optional[List[Optional[float]]] = None,
) -> pd.DataFrame: # index: formula, columns: adducts, values: exact masses of fragments
    formulas: pd.Series = pd.Series(formula)
    fragments = {"formula":formula}
    for adduct in adducts:
        if adduct == "M":
            fragments[adduct] = formulas.apply(lambda x: Fragment.from_string(x).ExactMass)
        else:
            fragments[adduct] = formulas.apply(lambda x: Fragment.from_string(x + adduct).ExactMass)
    if RT is not None:
        RT: pd.Series = pd.Series(RT)
        fragments['RT'] = RT
    fragments = pd.DataFrame(fragments)
    return fragments

def search_fragments_to_matrix(
    fragment_MZs: pd.DataFrame,
    ion_MZs: np.ndarray, # shape: (n_ions,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    fragment_RTs: Optional[pd.Series] = None,
    ion_RTs: Optional[np.ndarray] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
) -> np.ndarray: # shape: (n_ions, n_fragments, n_adducts)
    d_matrix = np.abs(fragment_MZs.values[np.newaxis,:] - ion_MZs[:,np.newaxis,np.newaxis])
    if mz_tolerance_type == 'ppm':
        d_matrix = d_matrix / fragment_MZs.values[np.newaxis,:] * 1e6
    I_F_D_matrix = d_matrix <= mz_tolerance
    if fragment_RTs is not None and ion_RTs is not None:
        d_matrix_RT = np.abs(fragment_RTs.values[np.newaxis,:] - ion_RTs[:,np.newaxis])
        bool_matrix_RT = d_matrix_RT <= RT_tolerance
        I_F_D_matrix = I_F_D_matrix & bool_matrix_RT[:, :, np.newaxis]
    F_D_matrix: np.ndarray = I_F_D_matrix.sum(axis=0)
    F_matrix: np.ndarray = F_D_matrix.sum(axis=1)
    F_matrix = F_matrix >= adduct_co_occurrence_threshold
    I_F_D_matrix = I_F_D_matrix & F_matrix[np.newaxis, :, np.newaxis]
    return I_F_D_matrix

def search_fragments(
    formulas: pd.Series,
    fragment_MZs: pd.DataFrame,
    ion_MZs: np.ndarray, # shape: (n_ions,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    fragment_RTs: Optional[pd.Series] = None,
    ion_RTs: Optional[np.ndarray] = None,
    RT_tolerance: float = 0.1,
    adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    return_matrix: bool = False,
) -> Union[
    List[Dict[str, str]], # List[Dict[formula, adduct]]
    Tuple[List[Dict[str, str]], np.ndarray],
]:
    bool_matrix = search_fragments_to_matrix(
        fragment_MZs, ion_MZs, mz_tolerance, mz_tolerance_type, 
        fragment_RTs, ion_RTs, RT_tolerance, 
        adduct_co_occurrence_threshold,
    )
    indices = np.argwhere(bool_matrix)
    results: Dict[int, Dict[str, str]] = {}
    for ion_index, formula_index, adduct_index in indices:
        if ion_index not in results:
            results[ion_index] = {}
        formula = formulas[formula_index]
        results[ion_index][formula] = fragment_MZs.columns[adduct_index]
    results = dict2list(results, len(ion_MZs), {})
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
    query_embeddings: np.ndarray, # shape: (n_query_items, n_dim)
    ref_embeddings: np.ndarray, # shape: (n_ref_items, n_dim)
) -> np.ndarray: # shape: (n_query_items, n_ref_items)
    score_matrix = np.dot(query_embeddings, ref_embeddings.T)
    score_matrix = score_matrix / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) * np.linalg.norm(ref_embeddings, axis=1, keepdims=True).T)
    return score_matrix

def search_embeddings_numpy(
    query_embeddings: np.ndarray, # shape: (n_query_items, n_dim)
    ref_embeddings: np.ndarray, # shape: (n_ref_items, n_dim)
    mask: Optional[np.ndarray] = None, # shape: (n_query_items,n_ref_items)
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    score_matrix = cosine_similarity(query_embeddings, ref_embeddings)
    if mask is not None:
        score_matrix = score_matrix * mask
    I = np.argsort(-score_matrix, axis=-1)[:, :top_k]
    S = np.take_along_axis(score_matrix, I, axis=1)
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
        df['mzs'] = df['mzs'].apply(lambda x: np.array(x))
    if intensities is not None:
        df['intensities'] = intensities
        df['intensities'] = df['intensities'].apply(lambda x: np.array(x))
    return df
    
def search_precursors_to_matrix(
    query_precursor_mzs: np.ndarray, # shape: (n_query_precursors,)
    ref_precursor_mzs: np.ndarray, # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    query_precursor_RTs: Optional[np.ndarray] = None,
    ref_precursor_RTs: Optional[np.ndarray] = None,
    RT_tolerance: float = 0.1,
) -> np.ndarray: # shape: (n_query_precursors, n_ref_precursors)
    d_matrix = np.abs(query_precursor_mzs[np.newaxis,:] - ref_precursor_mzs[:,np.newaxis])
    if mz_tolerance_type == 'ppm':
        d_matrix = d_matrix / query_precursor_mzs[np.newaxis,:] * 1e6
    Q_R_matrix = d_matrix <= mz_tolerance
    if query_precursor_RTs is not None and ref_precursor_RTs is not None:
        d_matrix_RT = np.abs(query_precursor_RTs[np.newaxis,:] - ref_precursor_RTs[:,np.newaxis])
        bool_matrix_RT = d_matrix_RT <= RT_tolerance
        Q_R_matrix = Q_R_matrix & bool_matrix_RT[:, :, np.newaxis]
    return Q_R_matrix

def search_precursors(
    query_precursor_mzs: np.ndarray, # shape: (n_query_precursors,)
    ref_precursor_mzs: np.ndarray, # shape: (n_ref_precursors,)
    mz_tolerance: float = 3,
    mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
    query_precursor_RTs: Optional[np.ndarray] = None,
    ref_precursor_RTs: Optional[np.ndarray] = None,
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
    indices = np.argwhere(bool_matrix)
    results: Dict[int, List[int]] = {}
    for query_index, ref_index in indices:
        if query_index not in results:
            results[query_index] = []
        results[query_index].append(ref_index)
    results = dict2list(results, len(query_precursor_mzs), [])
    if return_matrix:
        return results, bool_matrix
    return results