from __future__ import annotations
import numpy as np
import modin.pandas as pd
import faiss
import tools
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable

class FragLib:
    
    non_adduct_columns = frozenset(['formula', 'RT'])

    def __init__(
        self, 
        formula: List[str], # list of formulas
        RT: Optional[List[Optional[float]]] = None, 
        adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
    ):
        self.fragments = tools.infer_fragments_table(formula, adducts, RT)
        
    @property
    def formulas(self) -> pd.Series:
        return self.fragments['formula']
    
    @property
    def adducts(self):
        return self.fragments.columns[self.fragments.columns.map(lambda x: x not in self.non_adduct_columns)]
    
    @property
    def MZs(self):
        return self.fragments[self.adducts]
    
    @property
    def RTs(self) -> Optional[pd.Series]:
        if 'RT' in self.fragments.columns:
            return self.fragments['RT']
        else:
            return None
        
    def search_to_matrix(
        self,
        query_mzs: np.ndarray, # shape: (n_ions,)
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[np.ndarray] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    ) -> np.ndarray: # shape: (n_queries, n_formulas), dtype: bool
        results = tools.search_fragments_to_matrix(
            self.MZs,query_mzs,
            mz_tolerance,mz_tolerance_type,
            self.RTs,query_RTs,RT_tolerance,
            adduct_co_occurrence_threshold,
        )
        results:np.ndarray = results.sum(axis=-1) > 0
        return results
        
    def search(
        self,
        query_mzs: np.ndarray, # shape: (n_ions,)
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[np.ndarray] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    ) -> List[
        Dict[
            str, # formula
            str # adducts
        ]
    ]:
        results = tools.search_fragments(
            self.formulas,self.MZs,query_mzs,
            mz_tolerance,mz_tolerance_type,
            self.RTs,query_RTs,RT_tolerance,
            adduct_co_occurrence_threshold,
        )
        return results
    
    def to_bytes(self) -> bytes:
        return tools.to_pickle_bytes(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> FragLib:
        return tools.from_pickle_bytes(data)
    
    def to_pickle(self, path: str):
        tools.save_pickle(self, path)
    
    @classmethod
    def from_pickle(cls, path: str) -> FragLib:
        return tools.load_pickle(path)

class MolLib:
    
    def __init__(
        self,
        smiles: List[str], # list of SMILES strings
        RT: Optional[List[Optional[float]]] = None,
        adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
        embeddings: Dict[str, np.ndarray] = {}, # dict of embedding arrays, keyed by embedding name
    ):
        formulas = tools.smiles2formula(smiles)
        self.smiles = np.array(smiles)
        self.fragments = FragLib(formulas, RT, adducts)
        self.embeddings = embeddings
        # self.embeddings: Dict[str, faiss.Index] = {}
        # for name, array in embeddings.items():
        #     self.embeddings[name] = faiss.IndexFlatIP(array.shape[1])
        #     self.embeddings[name].add(array)
        
    @property
    def SMILES(self):
        return self.smiles
        
    @property
    def formulas(self):
        return self.fragments.formulas
    
    @property
    def adducts(self):
        return self.fragments.adducts
    
    @property
    def MZs(self):
        return self.fragments.MZs
    
    @property
    def RTs(self):
        return self.fragments.RTs
    
    # def mol_embedding(self, embedding_name: str) -> Optional[faiss.Index]:
    #     return self.embeddings.get(embedding_name, None)
    
    def mol_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        return self.embeddings.get(embedding_name, None)
    
    def search(
        self,
        embedding_name: str,
        query_embedding: np.ndarray, # shape: (n_queries, embedding_dim)
        query_mzs: Optional[np.ndarray] = None, # shape: (n_ions,)
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[np.ndarray] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
    ) -> Tuple[
        List[List[str]], # list of SMILES strings
        List[List[float]], # list of scores
    ]:
        mask = None
        if query_mzs is not None:
            mask = self.fragments.search_to_matrix(
                query_mzs,mz_tolerance,mz_tolerance_type,
                query_RTs,RT_tolerance,
                adduct_co_occurrence_threshold,
            )
        index,scores = tools.search_embeddings_numpy(
            query_embedding,
            self.mol_embedding(embedding_name),
            mask,top_k,
        )
        smiles = np.take_along_axis(self.SMILES[np.newaxis,:], index, axis=1)
        smiles,scores = smiles.tolist(),scores.tolist()
        return smiles, scores

class SpecLib:
    
    pass