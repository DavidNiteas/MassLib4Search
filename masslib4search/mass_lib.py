from __future__ import annotations
import numpy as np
import modin.pandas as pd
# import pandas as pd
from . import tools
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable

class FragLib:
    
    non_adduct_columns = frozenset(['formula', 'RT'])

    def __init__(
        self, 
        formula: Union[List[str], pd.Series] = None, # list of formulas
        RT: Optional[List[Optional[float]]] = None, 
        adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
    ):
        if formula is None:
            formula = []
        self.fragments = tools.infer_fragments_table(formula, adducts, RT)
        
    def __len__(self):
        return len(self.index)
    
    @property
    def index(self):
        return self.fragments.index
        
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
        return self.fragments.get('RT', None)
        
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
        return_matrix: bool = False, # if True, return a matrix of shape (n_queries, n_formulas), dtype: bool, indicating which formulas are present for each query
    ) -> Union[
        List[Dict[str, str]], # List[Dict[formula, adduct]]
        Tuple[List[Dict[str, str]], np.ndarray],
    ]:
        results = tools.search_fragments(
            self.formulas,self.MZs,query_mzs,
            mz_tolerance,mz_tolerance_type,
            self.RTs,query_RTs,RT_tolerance,
            adduct_co_occurrence_threshold,
            return_matrix,
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
        smiles: List[str] = None, # list of SMILES strings
        RT: Optional[List[Optional[float]]] = None,
        adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
        embeddings: Dict[str, np.ndarray] = {}, # dict of embedding arrays, keyed by embedding name
    ):
        if smiles is None:
            smiles = []
        self.smiles = pd.Series(smiles)
        formulas = tools.smiles2formulas(self.smiles).dropna()
        self.fragments = FragLib(formulas, RT, adducts)
        self.smiles = self.smiles[formulas.index]
        if len(embeddings) > 0:
            self.embeddings = pd.Series(embeddings).apply(lambda x: x[self.fragments.index])
        # self.embeddings: Dict[str, faiss.Index] = {}
        # for name, array in embeddings.items():
        #     self.embeddings[name] = faiss.IndexFlatIP(array.shape[1])
        #     self.embeddings[name].add(array)
        
    def __len__(self):
        return len(self.index)
    
    @property
    def index(self):
        return self.smiles.index
        
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
    ) -> Dict[
        Literal['index','smiles','score','adduct'], # if query_mzs is None, adduct will not be included in the result
        Union[
            List[List[int]], # list of indices
            List[List[str]], # list of SMILES strings and adducts
            List[List[float]], # list of scores
        ]
    ]:
        mask = None
        adduct_map = None
        if query_mzs is not None:
            adduct_map,matrix = self.fragments.search(
                query_mzs,mz_tolerance,mz_tolerance_type,
                query_RTs,RT_tolerance,
                adduct_co_occurrence_threshold,
                return_matrix=True,
            )
            mask = matrix.sum(axis=-1) > 0
            
        index,scores = tools.search_embeddings_numpy(
            query_embedding,
            self.mol_embedding(embedding_name),
            mask,top_k,
        )
        smiles = np.take_along_axis(self.SMILES.values[np.newaxis,:], index, axis=1)
        results = {
            'index': index.tolist(),
            'smiles': smiles.tolist(),
            'score': scores.tolist(),
        }
        if adduct_map is not None:
            adduct_list: List[List[str]] = []
            formula_array = np.take_along_axis(self.fragments.formulas.values[np.newaxis,:], index, axis=1)
            for formulas,adducts in zip(formula_array,adduct_map):
                adduct_list.append([])
                for formula in formulas:
                    adduct_list[-1].append(adducts[formula])
            results['adduct'] = adduct_list
        return results
    
    def to_bytes(self) -> bytes:
        return tools.to_pickle_bytes(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> MolLib:
        return tools.from_pickle_bytes(data)
    
    def to_pickle(self, path: str):
        tools.save_pickle(self, path)
    
    @classmethod
    def from_pickle(cls, path: str) -> MolLib:
        return tools.load_pickle(path)

class SpecLib:
    
    def __init__(
        self,
        PI: Optional[List[float]] = None,
        RT: Optional[List[float]] = None,
        mzs: Optional[List[List[float]]] = None,
        intensities: Optional[List[List[float]]] = None,
        spec_embeddings: Dict[str, np.ndarray] = {}, # dict of embedding arrays, keyed by embedding name
        smiles: List[str] = None, # list of SMILES strings
        mol_RT: Optional[List[Optional[float]]] = None,
        mol_adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'], # list of adducts for each SMILES string
        mol_embeddings: Dict[str, np.ndarray] = {}, # dict of embedding arrays, keyed by embedding name
    ):
        self.precursors = tools.infer_precursors_table(PI, RT)
        self.peaks = tools.infer_peaks_table(mzs, intensities)
        self.spec_embeddings = pd.Series(spec_embeddings)
        self.mols = MolLib(smiles, mol_RT, mol_adducts, mol_embeddings)
        if len(self.MolLib) > 0:
            if len(self.precursors) > 0:
                self.precursors = self.precursors.loc[self.MolLib.index]
            if len(self.peaks) > 0:
                self.peaks = self.peaks.loc[self.MolLib.index]
            if len(self.spec_embeddings) > 0:
                self.spec_embeddings = self.spec_embeddings.apply(lambda x: x[self.MolLib.index])
        
    def __len__(self):
        return max(
            len(self.precursors), 
            len(self.peaks), 
            len(self.mols),
            self.spec_embeddings.apply(lambda x: len(x)).max(),
        )
        
    @property
    def PIs(self) -> Optional[pd.Series]:
        return self.precursors.get('PI', None)
    
    @property
    def RTs(self) -> Optional[pd.Series]:
        return self.precursors.get('RT', None)
    
    @property
    def mzs(self) -> Optional[pd.Series]:
        return self.peaks.get('mz', None)
    
    @property
    def intensities(self) -> Optional[pd.Series]:
        return self.peaks.get('intensity', None)
    
    @property
    def precursor_mzs(self) -> Optional[pd.Series]:
        return self.precursors.get('mz', None)
    
    @property
    def precursor_intensities(self) -> Optional[pd.Series]:
        return self.precursors.get('intensity', None)
    
    @property
    def MolLib(self) -> MolLib:
        return self.mols
    
    @property
    def mol_smiles(self) -> pd.Series:
        return self.MolLib.SMILES
    
    @property
    def mol_adducts(self):
        return self.MolLib.adducts
    
    @property
    def mol_MZs(self):
        return self.MolLib.MZs
    
    @property
    def mol_RTs(self):
        return self.MolLib.RTs
    
    @property
    def mol_formulas(self):
        return self.MolLib.formulas
    
    def spec_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        return self.spec_embeddings.get(embedding_name, None)
    
    def mol_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        return self.mols.mol_embedding(embedding_name)
    
    def search_precursors_to_matrix(
        self,
        query_precursor_mzs: np.ndarray, # shape: (n_queries,)
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_precursor_RTs: Optional[np.ndarray] = None,
        precursor_RT_tolerance: float = 0.1,
    ) -> np.ndarray: # shape: (n_queries, n_formulas), dtype: bool
        results = tools.search_precursors_to_matrix(
            query_precursor_mzs,
            self.precursor_mzs.values,
            precursor_mz_tolerance,
            precursor_mz_tolerance_type,
            query_precursor_RTs,
            self.RTs.values,
            precursor_RT_tolerance,
        )
        return results
    
    def search_precursors(
        self,
        query_precursor_mzs: np.ndarray, # shape: (n_queries,)
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_precursor_RTs: Optional[np.ndarray] = None,
        precursor_RT_tolerance: float = 0.1,
        return_matrix: bool = False, # if True, return a matrix of shape (n_queries, n_formulas), dtype: bool, indicating which formulas are present for each query
    ) -> Union[
        List[List[int]], # List[List[ref_precursor_index]]
        Tuple[List[List[int]], np.ndarray],
    ]:
        results = tools.search_precursors(
            query_precursor_mzs,
            self.precursor_mzs.values,
            precursor_mz_tolerance,
            precursor_mz_tolerance_type,
            query_precursor_RTs,
            self.RTs.values,
            precursor_RT_tolerance,
            return_matrix,
        )
        return results
    
    def search_embedding(
        self,
        embedding_name: str,
        query_embedding: np.ndarray, # shape: (n_queries, embedding_dim)
        query_precursor_mzs: Optional[np.ndarray] = None, # shape: (n_queries,)
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
        precursor_search_type: Literal['mol', 'spec'] = 'mol',
    ) -> Dict[
        Literal['index','smiles','score','adduct'],
        Union[
            List[List[int]], # list of indices
            List[List[str]], # list of SMILES strings and adducts
            List[List[float]], # list of scores
        ]
    ]:
        mask = None
        adduct_map = None
        if query_precursor_mzs is not None and len(self.MolLib) > 0 and precursor_search_type == 'mol':
            adduct_map,matrix = self.MolLib.fragments.search(
                query_mzs=query_precursor_mzs,
                mz_tolerance=precursor_mz_tolerance,
                mz_tolerance_type=precursor_mz_tolerance_type,
                adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
                return_matrix=True,
            )
            mask = matrix.sum(axis=-1) > 0
        if query_precursor_mzs is not None and len(self.precursors) > 0 and precursor_search_type =='spec':
            matrix = self.search_precursors_to_matrix(
                query_precursor_mzs,
                precursor_mz_tolerance,
                precursor_mz_tolerance_type,
            )
            mask = matrix.sum(axis=-1) > 0
        
        ref_embedding = self.spec_embedding(embedding_name)
        if ref_embedding is None:
            raise ValueError(f"No embedding named {embedding_name} found in SpecLib.")
        
        index,scores = tools.search_embeddings_numpy(
            query_embedding,
            ref_embedding,
            mask,top_k,
        )
        smiles = np.take_along_axis(self.mol_smiles[np.newaxis,:], index, axis=1)
        results = {
            'index': index.tolist(),
            'smiles': smiles.tolist(),
            'score': scores.tolist(),
        }
        if adduct_map is not None:
            adduct_list: List[List[str]] = []
            formula_array = np.take_along_axis(self.MolLib.fragments.formulas.values[np.newaxis,:], index, axis=1)
            for formulas,adducts in zip(formula_array,adduct_map):
                adduct_list.append([])
                for formula in formulas:
                    adduct_list[-1].append(adducts[formula])
            results['adduct'] = adduct_list
        return results
    
    def to_bytes(self) -> bytes:
        return tools.to_pickle_bytes(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SpecLib:
        return tools.from_pickle_bytes(data)
    
    def to_pickle(self, path: str):
        tools.save_pickle(self, path)
    
    @classmethod
    def from_pickle(cls, path: str) -> SpecLib:
        return tools.load_pickle(path)
    
class SpecRuleLib:
    
    pass