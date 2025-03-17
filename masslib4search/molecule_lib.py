from __future__ import annotations
from .mass_lib_utils import BaseLib,Molecules,Embeddings
from .fragment_lib import FragLib
from . import search_tools,base_tools
import dask
import dask.bag as db
import dask.array as da
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

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

class MolLib(BaseLib):
    
    @staticmethod
    def lazy_init(
        smiles: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        embeddings: Dict[str, Union[NDArray[np.float32],db.Bag,da.Array]] = {},
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        metadatas: Optional[pd.DataFrame] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['index', 'mols', 'embeddings', 'fragments',],
        Union[dict,
        pd.Index,Sequence[Hashable],
        pd.DataFrame, # metadatas
        None,]
    ]:
        mols_lazy_dict = Molecules.lazy_init(smiles, index=index, npartitions=npartitions)
        embeddings_lazy_dict = Embeddings.lazy_init(embeddings, index=index, npartitions=npartitions)
        if not isinstance(smiles, db.Bag):
            smiles = db.from_sequence(smiles,npartitions=npartitions)
        formulas = smiles.map(lambda x: smiles2formula(x))
        frag_lib_lazy_dict = FragLib.lazy_init(formulas, RT, adducts, npartitions)
        return {
            'index': index,
            'mols': mols_lazy_dict,
            'embeddings': embeddings_lazy_dict,
            'fragments': frag_lib_lazy_dict,
            'metadatas': metadatas,
        }
        
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        computed_lazy_dict['mols'] = Molecules(computed_lazy_dict=computed_lazy_dict['mols'])
        computed_lazy_dict['fragments'] = FragLib(computed_lazy_dict=computed_lazy_dict['fragments'])
        computed_lazy_dict['embeddings'] = Embeddings(computed_lazy_dict=computed_lazy_dict['embeddings'])
        self.index = self.get_index(computed_lazy_dict)
        self.molecules = computed_lazy_dict['mols']
        self.fragments = computed_lazy_dict['fragments']
        self.embeddings = computed_lazy_dict['embeddings']
        self.metadatas = computed_lazy_dict['metadatas']
    
    def __init__(
        self,
        smiles: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        embeddings: Dict[str, Union[NDArray[np.float32],db.Bag,da.Array]] = {},
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        metadatas: Optional[pd.DataFrame] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not (all(x is None for x in [smiles,RT,index,computed_lazy_dict]) and len(embeddings) == 0):
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(smiles, RT, adducts, embeddings, index, metadatas, num_workers)
                (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
            self.from_lazy(**computed_lazy_dict)
    
    @property
    def Index(self):
        return self.index
        
    @property
    def SMILES(self):
        return self.molecules.SMILES
    
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
    def Metadatas(self):
        return self.metadatas
    
    @property
    def bad_index(self) -> pd.Index:
        return self.FragmentLib.bad_index
    
    def get_embedding(
        self,
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> pd.Series:
        return self.embeddings.get_embedding(embedding_name,i_and_key)
    
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
        if self.FragmentLib.is_empty or query_mzs is None:
            return None
        else:
            return self.FragmentLib.search_fragments(
                query_mzs,
                adducts,
                mz_tolerance,  
                mz_tolerance_type,
                query_RTs,
                RT_tolerance,
                adduct_co_occurrence_threshold,  
                batch_size,
            )
    
    def search_embedding(
        self,
        embedding_name: str,
        query_embedding: pd.Series, # shape: (n_queries, embedding_dim)
        query_mzs: pd.Series = None, # shape: (n_ions,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[NDArray[np.float_]] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
        search_type: Literal['mol'] = 'mol', # MolLib only supports 'mol' search type
        batch_size: int = 10,
    ) -> pd.DataFrame: # columns: db_index, smiles, score, formula, adduct
        
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
            tag_ref_index = fragment_results['db_ids']
        ref_embedding = self.get_embedding(embedding_name)
        embedding_results = search_tools.search_embeddings(
            query_embedding,ref_embedding,tag_ref_index,top_k
        )
        
        print('Merging results...')
        qry_index_bag = db.from_sequence(embedding_results.index,partition_size=1)
        smiles_bag = qry_index_bag.map(lambda x: self.molecules.SMILES[embedding_results.loc[x,'db_ids']].values if not isinstance(embedding_results.loc[x,'db_ids'],str) else 'null')
        if fragment_results is not None:
            
            def get_sub_index(x: Hashable) -> Tuple[Hashable,Union[NDArray[np.int64],Literal['null']]]:
                embedding_tag_index:NDArray[np.int64] = embedding_results.loc[x,'db_ids']
                fragment_tag_index:NDArray[np.int64] = fragment_results.loc[x,'db_ids']
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
        
        return pd.DataFrame({'db_ids': embedding_results['db_ids'].tolist(), 'smiles': smiles, 'score': embedding_results['score'].tolist(), 'formula': formula, 'adduct': adduct},index=query_embedding.index)
    
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> MolLib:
        if self.is_empty:
            return self.__class__()
        iloc = self.format_selection(i_and_key)
        new_lib = MolLib()
        new_lib.molecules = self.item_select(self.molecules, iloc)
        new_lib.fragments = self.item_select(self.fragments, iloc)
        new_lib.embeddings = self.item_select(self.embeddings, iloc)
        new_lib.metadatas = self.item_select(self.metadatas, iloc)
        new_lib.index = self.Index[iloc]
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> MolLib:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> MolLib:
        return base_tools.load_pickle(path)
    
    def to_dataframes(self):
        dataframes = {
            '/MolLib/index': pd.DataFrame({'index': self.index},index=self.index),
            '/MolLib/mols': self.molecules.to_dataframe()
        }
        fragments_dfs = self.fragments.to_dataframes()
        for k,v in fragments_dfs.items():
            dataframes[f'/MolLib{k}'] = v
        if not self.embeddings.is_empty:
            dataframes['/MolLib/embeddings'] = self.embeddings.to_dataframe()
        if self.metadatas is not None:
            dataframes['/MolLib/metadatas'] = self.metadatas
        return dataframes
    
    @classmethod
    def from_dataframes(cls, dataframes: Dict[str,pd.DataFrame]) -> MolLib:
        new_lib = MolLib()
        new_lib.embeddings = Embeddings()
        new_lib.index = None
        new_lib.metadatas = None
        new_lib.molecules = None
        new_lib.fragments = FragLib.from_dataframes(dataframes)
        for key in dataframes:
            if key.endswith('/MolLib/mols'):
                new_lib.molecules = Molecules.from_dataframe(dataframes[key])
            elif key.endswith('/MolLib/embeddings'):
                new_lib.embeddings = Embeddings.from_dataframe(dataframes[key])
            elif key.endswith('/MolLib/metadatas'):
                new_lib.metadatas = dataframes[key]
            elif key.endswith('/MolLib/index'):
                new_lib.index = dataframes[key].index