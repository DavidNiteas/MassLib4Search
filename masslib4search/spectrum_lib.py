from __future__ import annotations
from . import base_tools,search_tools
from .mass_lib_utils import BaseLib,Spectrums,Embeddings
from .molecule_lib import MolLib,FragLib
import dask
import dask.bag as db
import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

class SpecLib(BaseLib):
    
    @staticmethod
    def lazy_init(
        PI: Union[List[float],db.Bag,None] = None,
        RT: Union[List[float],db.Bag,None] = None,
        mzs: Union[List[List[float]],db.Bag,None] = None,
        intensities: Union[List[List[float]],db.Bag,None] = None,
        spec_embeddings: Dict[str, Union[NDArray[np.float32],da.Array]] = {},
        smiles: Union[List[str],db.Bag,None] = None,
        mol_RT: Union[List[Optional[float]],db.Bag,None] = None,
        mol_adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        mol_embeddings: Dict[str, Union[NDArray[np.float32],da.Array]] = {},
        index: Union[pd.Index,db.Bag,None] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['index', 'spec', 'spec_embeddings', 'mol_lib'],
        Union[
            db.Bag,
            dict, # spec_embeddings, mol_lib_lazy_dict, spec
            pd.Index,Sequence[Hashable], # index
            None,
        ]
    ]:
        spec_lazy_dict = Spectrums.lazy_init(PI, RT, mzs, intensities, index=index, npartitions=npartitions)
        spec_embdeddings_lazy_dict = Embeddings.lazy_init(spec_embeddings, index=index, npartitions=npartitions)
        mol_lib_lazy_dict = MolLib.lazy_init(smiles, mol_RT, mol_adducts, mol_embeddings, index=index, npartitions=npartitions)
        return {
            'index': index,
            'spec': spec_lazy_dict,
            'spec_embeddings': spec_embdeddings_lazy_dict,
            'mol_lib': mol_lib_lazy_dict,
        }
        
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        computed_lazy_dict['spec'] = Spectrums(computed_lazy_dict=computed_lazy_dict['spec'])
        computed_lazy_dict['spec_embeddings'] = Embeddings(computed_lazy_dict=computed_lazy_dict['spec_embeddings'])
        computed_lazy_dict['mol_lib'] = MolLib(computed_lazy_dict=computed_lazy_dict['mol_lib'])
        self.index = self.get_index(computed_lazy_dict)
        self.spectrums = computed_lazy_dict['spec']
        self.spec_embeddings = computed_lazy_dict['spec_embeddings']
        self.mol_lib = computed_lazy_dict['mol_lib']
    
    def __init__(
        self,
        PI: Optional[List[float]] = None,
        RT: Optional[List[float]] = None,
        mzs: Optional[List[List[float]]] = None,
        intensities: Optional[List[List[float]]] = None,
        spec_embeddings: Dict[str, NDArray[np.float32]] = {}, # dict of embedding arrays, keyed by embedding name
        smiles: List[str] = None, # list of SMILES strings
        mol_RT: Optional[List[Optional[float]]] = None,
        mol_adducts: Union[List[str],Literal['pos','neg']] = 'pos', # list of adducts for each SMILES string
        mol_embeddings: Dict[str, NDArray[np.float32]] = {}, # dict of embedding arrays, keyed by embedding name
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not \
            (all(x is None for x in [PI, RT, mzs, intensities, smiles, mol_RT, index, computed_lazy_dict]) \
            and all(len(x) == 0 for x in [spec_embeddings, mol_embeddings])):
                
            if not isinstance(computed_lazy_dict,dict):
                lazy_dict = self.lazy_init(PI, RT, mzs, intensities, spec_embeddings, smiles, mol_RT, mol_adducts, mol_embeddings, index, num_workers)
                (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
            self.from_lazy(**computed_lazy_dict)
        
    @property
    def Index(self):
        return self.index
        
    @property
    def PIs(self) -> Optional[pd.Series]:
        return self.spectrums.PIs
    
    @property
    def RTs(self) -> Optional[pd.Series]:
        return self.spectrums.RTs
    
    @property
    def MZs(self) -> Optional[pd.Series]:
        return self.spectrums.MZs
    
    @property
    def Intens(self) -> Optional[pd.Series]:
        return self.spectrums.Intens
    
    @property
    def MolLib(self) -> MolLib:
        return self.mol_lib
    
    @property
    def FragmentLib(self) -> FragLib:
        return self.MolLib.FragmentLib
    
    @property
    def MolSMILES(self) -> pd.Series:
        return self.MolLib.SMILES
    
    @property
    def MolAdducts(self):
        return self.MolLib.Adducts
    
    @property
    def MolMZs(self):
        return self.MolLib.MZs
    
    @property
    def MolRTs(self):
        return self.MolLib.RTs
    
    @property
    def MolFormulas(self):
        return self.MolLib.Formulas
    
    def get_spec_embedding(
        self, 
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> NDArray[np.float32]:
        return self.spec_embeddings.get_embedding(embedding_name, i_and_key)
    
    def get_mol_embedding(
        self,
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> NDArray[np.float32]:
        return self.MolLib.get_embedding(embedding_name, i_and_key)
    
    def search_fragments(
        self,
        query_mzs: pd.Series,  # Series[float], shape: (n_ions,)
        adducts: Union[List[str], Literal['all_adducts', 'no_adducts']] = 'all_adducts',  # 加合物选择模式，'all_adducts' 表示考虑所有加合物（不包括 [M]），'no_adducts' 表示只考虑 [M]
        mz_tolerance: float = 3,  
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[NDArray[np.float_]] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1,  # if a formula has less than this number of adducts, it will be removed from the result
        batch_size: int = 10,
    ) -> Optional[pd.DataFrame]:
        if self.MolLib.is_empty or query_mzs is None:
            return None
        else:
            return self.MolLib.search_fragments(
                query_mzs,
                adducts,
                mz_tolerance,  
                mz_tolerance_type,
                query_RTs,
                RT_tolerance,
                adduct_co_occurrence_threshold,  
                batch_size,
            )
            
    def search_precursors(
        self,
        query_mzs: pd.Series,  # Series[float], shape: (n_ions,)
        mz_tolerance: float = 3,  
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[NDArray[np.float_]] = None,
        RT_tolerance: float = 0.1,
        batch_size: int = 10,
    ) -> Optional[pd.DataFrame]:
        if self.PIs is None or query_mzs is None:
            return None
        else:
            return search_tools.search_precursors(
                query_mzs,self.PIs,
                mz_tolerance,mz_tolerance_type,
                self.RTs,query_RTs,RT_tolerance,
                batch_size
            )
    
    def search_embedding(
        self,
        embedding_name: str,
        query_embedding: pd.Series, # shape: (n_queries, embedding_dim)
        query_mzs: pd.Series = None, # shape: (n_queries,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[NDArray[np.float_]] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
        search_type: Literal['mol', 'spec'] = 'mol',
        batch_size: int = 10,
    ) -> pd.DataFrame: # columns: db_index, smiles, score, formula, adduct
        
        if search_type == 'mol':
            
            fragment_results = self.search_fragments(
                                    query_mzs,
                                    adducts,
                                    mz_tolerance,
                                    mz_tolerance_type,
                                    query_RTs,
                                    RT_tolerance,
                                    adduct_co_occurrence_threshold,
                                    batch_size
            )
        
        else:
            
            fragment_results = self.search_fragments(
                                    query_mzs,
                                    adducts,
                                    mz_tolerance,
                                    mz_tolerance_type,
                                    query_RTs,
                                    RT_tolerance,
                                    adduct_co_occurrence_threshold,
                                    batch_size
            )
            
        if fragment_results is None:
            tag_ref_index = None
        else:
            tag_ref_index = fragment_results['db_index']
            
        ref_embedding = self.get_spec_embedding(embedding_name)
        embedding_results = search_tools.search_embeddings(
            query_embedding,ref_embedding,tag_ref_index,top_k
        )
        
        print('Merging results...')
        qry_index_bag = db.from_sequence(embedding_results.index,partition_size=1)
        smiles_bag = qry_index_bag.map(lambda x: self.MolLib.SMILES[embedding_results.loc[x,'db_index']].values if not isinstance(embedding_results.loc[x,'db_index'],str) else 'null')
        if fragment_results is not None:
            
            def get_sub_index(x: Hashable) -> Tuple[Hashable,Union[NDArray[np.int64],Literal['null']]]:
                embedding_tag_index:NDArray[np.int64] = embedding_results.loc[x,'db_index']
                fragment_tag_index:NDArray[np.int64] = fragment_results.loc[x,'db_index']
                if isinstance(embedding_tag_index, np.ndarray) and isinstance(fragment_tag_index, np.ndarray):
                    return x,np.where(embedding_tag_index[:,np.newaxis] == fragment_tag_index[np.newaxis,:])[1]
                else:
                    return x,'null'
                
            db_index_iloc_bag = qry_index_bag.map(get_sub_index)
            formula_bag = db_index_iloc_bag.map(lambda x: fragment_results.loc[x[0],'formula'][x[1]] if not isinstance(x[1],str) else 'null')
            adduct_bag = db_index_iloc_bag.map(lambda x: fragment_results.loc[x[0],'adduct'][x[1]] if not isinstance(x[1],str) else 'null')

        else:
            formula_bag = None
            adduct_bag = None
        
        smiles,formula,adduct = dask.compute(smiles_bag,formula_bag,adduct_bag,scheduler='threads')
        
        return pd.DataFrame({'db_index': embedding_results['db_index'].tolist(), 'smiles': smiles, 'score': embedding_results['score'].tolist(), 'formula': formula, 'adduct': adduct},index=query_embedding.index)
    
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> SpecLib:
        if self.is_empty:
            return self.__class__()
        iloc = self.format_selection(i_and_key)
        new_lib = SpecLib()
        new_lib.spectrums = self.item_select(self.spectrums,iloc)
        new_lib.spec_embeddings = self.item_select(self.spec_embeddings,iloc)
        new_lib.mol_lib = self.item_select(self.mol_lib,iloc)
        new_lib.index = self.Index[iloc]
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SpecLib:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> SpecLib:
        return base_tools.load_pickle(path)