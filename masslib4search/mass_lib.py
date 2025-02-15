from __future__ import annotations
import dask
import dask.bag as db
import dask.array as da
import numpy as np
import pandas as pd
from . import tools
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable

# dev import
import sys
sys.path.append('.')
from MZInferrer.mzinferrer.mz_infer_tools import Fragment

class FragLib:
    
    non_adduct_columns = frozenset(['formula', 'RT'])
    
    @staticmethod
    def lazy_init(
        formula: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: List[str] = ['M','H+', 'NH4+', 'Na+', 'K+'],
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['formula', 'RT', 'adducts'],
        Union[
            db.Bag,
            List[str], # Adducts
            List[Optional[float]], #RT
        ]
    ]:
        if not isinstance(formula, db.Bag):
            formula = db.from_sequence(formula,npartitions=npartitions)
        lazy_dict = {"formula": formula, 'adducts': adducts}
        for adduct in adducts:
            if adduct == "M":
                lazy_dict[adduct] = formula.map(lambda x: Fragment.from_string(x).ExactMass if x is not None else None)
            else:
                lazy_dict[adduct] = formula.map(lambda x: Fragment.from_string(x + adduct).ExactMass if x is not None else None)
        if RT is not None:
            lazy_dict['RT'] = RT
        return lazy_dict
    
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.fragments = {
            "formula": computed_lazy_dict["formula"],
        }
        for adduct in computed_lazy_dict['adducts']:
            self.fragments[adduct] = computed_lazy_dict[adduct]
        if 'RT' in computed_lazy_dict:
            self.fragments['RT'] = computed_lazy_dict['RT']
        self.fragments = pd.DataFrame(self.fragments)

    def __init__(
        self, 
        formula: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: List[str] = ['M','H+', 'NH4+', 'Na+', 'K+'],
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if formula is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(formula, RT, adducts)
                (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
            self.from_lazy(**computed_lazy_dict)
        
    def __len__(self):
        return len(self.index)
    
    @property
    def index(self):
        return self.Fragments.index
    
    @property
    def Fragments(self) -> pd.DataFrame:
        return self.fragments
        
    @property
    def Formulas(self) -> pd.Series:
        return self.Fragments['formula']
    
    @property
    def Adducts(self):
        return self.Fragments.columns[self.Fragments.columns.map(lambda x: x not in self.non_adduct_columns)]
    
    @property
    def MZs(self):
        return self.Fragments[self.Adducts]
    
    @property
    def RTs(self) -> Optional[pd.Series]:
        return self.Fragments.get('RT', None)
    
    @property
    def bad_index(self) -> pd.Index:
        return self.index[self.Formulas.isna()]
        
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
            self.Formulas,self.MZs,query_mzs,
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
    
    @staticmethod
    def lazy_init(
        smiles: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
        embeddings: Dict[str, Union[np.ndarray,db.Bag,da.Array]] = {},
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['smiles', 'embeddings', 'fragments'],
        Union[
            db.Bag,
            List[str], # Adducts
            List[Optional[float]], #RT
        ]
    ]:
        if not isinstance(smiles, db.Bag):
            smiles = db.from_sequence(smiles,npartitions=npartitions)
        formulas = smiles.map(lambda x: tools.smiles2formula(x))
        frag_lib_lazy_dict = FragLib.lazy_init(formulas, RT, adducts, npartitions)
        return {
            'smiles': smiles,
            'embeddings': embeddings,
            'fragments': frag_lib_lazy_dict,
        }
        
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.smiles = pd.Series(computed_lazy_dict['smiles'])
        self.fragments = FragLib(computed_lazy_dict=computed_lazy_dict['fragments'])
        self.embeddings = pd.Series(computed_lazy_dict['embeddings'])
    
    def __init__(
        self,
        smiles: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
        embeddings: Dict[str, Union[np.ndarray,db.Bag,da.Array]] = {},
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if smiles is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(smiles, RT, adducts, embeddings)
                (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
            self.from_lazy(**computed_lazy_dict)
        
    def __len__(self):
        return len(self.index)
    
    @property
    def index(self):
        return self.SMILES.index
        
    @property
    def SMILES(self):
        return self.smiles
    
    @property
    def FragmentLib(self) -> FragLib:
        return self.fragments
    
    @property
    def Fragments(self) -> pd.DataFrame:
        return self.FragmentLib.Fragments
        
    @property
    def Formulas(self):
        return self.FragmentLib.Formulas
    
    @property
    def Adducts(self):
        return self.FragmentLib.Adducts
    
    @property
    def MZs(self):
        return self.FragmentLib.MZs
    
    @property
    def RTs(self):
        return self.FragmentLib.RTs
    
    @property
    def bad_index(self) -> pd.Index:
        return self.FragmentLib.bad_index
    
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
            formula_array = np.take_along_axis(self.Formulas.values[np.newaxis,:], index, axis=1)
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
    
    @staticmethod
    def lazy_init(
        PI: Union[List[float],db.Bag,None] = None,
        RT: Union[List[float],db.Bag,None] = None,
        mzs: Union[List[List[float]],db.Bag,None] = None,
        intensities: Union[List[List[float]],db.Bag,None] = None,
        spec_embeddings: Dict[str, Union[np.ndarray,da.Array]] = {},
        smiles: Union[List[str],db.Bag,None] = None,
        mol_RT: Union[List[Optional[float]],db.Bag,None] = None,
        mol_adducts: List[str] = ['H+', 'NH4+', 'Na+', 'K+'],
        mol_embeddings: Dict[str, Union[np.ndarray,da.Array]] = {},
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['PI', 'RT','mzs', 'intensities','spec_embeddings','mol_lib'],
        Union[
            db.Bag,
            List[float], # PI, RT
            List[List[float]], # mzs, intensities
            dict, # spec_embeddings, mol_lib_lazy_dict
        ]
    ]:
        if mzs is not None and not isinstance(mzs, db.Bag):
            mzs = db.from_sequence(mzs,npartitions=npartitions)
        if intensities is not None and not isinstance(intensities, db.Bag):
            intensities = db.from_sequence(intensities,npartitions=npartitions)
        if isinstance(mzs, db.Bag):
            mzs = mzs.map(lambda x: np.array(x))
        if isinstance(intensities, db.Bag):
            intensities = intensities.map(lambda x: np.array(x))
        mol_lib_lazy_dict = MolLib.lazy_init(smiles, mol_RT, mol_adducts, mol_embeddings, npartitions)
        return {
            'PI': PI,
            'RT': RT,
            'mzs': mzs,
            'intensities': intensities,
            'spec_embeddings': spec_embeddings,
            'mol_lib': mol_lib_lazy_dict,
        }
        
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.precursors = tools.infer_precursors_table(computed_lazy_dict['PI'], computed_lazy_dict['RT'])
        self.peaks = tools.infer_peaks_table(computed_lazy_dict['mzs'], computed_lazy_dict['intensities'])
        self.spec_embeddings = pd.Series(computed_lazy_dict['spec_embeddings'])
        self.mols = MolLib(computed_lazy_dict=computed_lazy_dict['mol_lib'])
    
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
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not isinstance(computed_lazy_dict,dict):
            lazy_dict = self.lazy_init(PI, RT, mzs, intensities, spec_embeddings, smiles, mol_RT, mol_adducts, mol_embeddings)
            (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
        self.from_lazy(**computed_lazy_dict)
        
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
        return self.MolLib.Adducts
    
    @property
    def mol_MZs(self):
        return self.MolLib.MZs
    
    @property
    def mol_RTs(self):
        return self.MolLib.RTs
    
    @property
    def mol_formulas(self):
        return self.MolLib.Formulas
    
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
            formula_array = np.take_along_axis(self.MolLib.Formulas.values[np.newaxis,:], index, axis=1)
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