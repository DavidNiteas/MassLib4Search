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
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['index', 'mols', 'embeddings', 'fragments',],
        dict,
        pd.Index,Sequence[Hashable],
        None,
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
    
    def __init__(
        self,
        smiles: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        embeddings: Dict[str, Union[NDArray[np.float32],db.Bag,da.Array]] = {},
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if smiles is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(smiles, RT, adducts, embeddings, index, num_workers)
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
    def bad_index(self) -> pd.Index:
        return self.FragmentLib.bad_index
    
    def mol_embedding_array(
        self,
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> NDArray[np.float32]:
        return self.embeddings.get_embedding_array(embedding_name,i_and_key)
    
    def search_embedding(
        self,
        embedding_name: str,
        query_embedding: Union[NDArray[np.float32],da.Array], # shape: (n_queries, embedding_dim)
        query_mzs: Union[NDArray[np.float_],da.Array,None] = None, # shape: (n_ions,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Union[NDArray[np.float_],da.Array,None] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
    ) -> Dict[
        Literal['index','smiles','score','formula','adduct'],
        Union[
            List[NDArray[np.int_]], # index of hits
            List[NDArray[np.str_]], # smiles/formula/adduct of hits
            List[NDArray[np.float_]], # scores of hits
        ]
    ]:
        '''
        根据给定的嵌入名称和查询嵌入，在分子库中搜索与查询离子匹配的分子。匹配基于质荷比(mz)和保留时间(RT)的容差值。
        函数可以返回匹配项的索引、smiles、分数，以及可选的匹配项的formula和adduct。

        参数:
        - embedding_name: 要使用的嵌入名称。
        - query_embedding: 查询嵌入的数组，可以是numpy数组或dask数组，形状为(n_queries, embedding_dim)。
        - query_mzs: 查询离子的质荷比数组，可以是numpy数组或dask数组，形状为(n_ions,)，可选。
        - adducts: 要考虑的加成型列表或字符串，字符串可以是'all_adducts'或'no_adducts'，可选。
        - mz_tolerance: 质荷比容差值，默认为3。
        - mz_tolerance_type: 质荷比容差类型，默认为'ppm'。
        - query_RTs: 查询离子的保留时间数组，可以是numpy数组或dask数组，形状为(n_ions,)，可选。
        - RT_tolerance: 保留时间容差值，默认为0.1分钟。
        - adduct_co_occurrence_threshold: 如果某个公式的共现加成型少于这个阈值，则从结果中移除该公式，默认为1。
        - top_k: 对于每个查询返回的匹配项数量，默认为5。

        返回值:
        - 一个字典，包含匹配项的索引、smiles、分数，以及可选的formula和adduct。
        - 字典中的键:
            - 'index': 匹配项的索引，列表长度为查询项数量，每个元素为一个numpy数组，形状为(tag_formula_num,)。
            - 'smiles': 匹配项的smiles，列表长度为查询项数量，每个元素为一个numpy数组，形状为(tag_formula_num,)。
            - 'score': 匹配项的分数，列表长度为查询项数量，每个元素为一个numpy数组，形状为(tag_formula_num,)。
            - 'formula': 匹配项的formula，仅在map_adducts为True时返回，列表长度为查询项数量，每个元素为一个numpy数组，形状为(tag_formula_num,)。
            - 'adduct': 匹配项的adduct，仅在map_adducts为True时返回，列表长度为查询项数量，每个元素为一个numpy数组，形状为(tag_formula_num,)。
        '''
        
        bool_matrix = None
        mask = None
        map_adducts = False
        
        if query_mzs is not None:
            
            if isinstance(adducts, str):
                if adducts == 'all_adducts':
                    adducts = self.Adducts
                elif adducts == 'no_adducts':
                    adducts = ['[M]']
            
            bool_matrix = self.FragmentLib.search_to_matrix(
                query_mzs,adducts,
                mz_tolerance,mz_tolerance_type,
                query_RTs,RT_tolerance,
                adduct_co_occurrence_threshold,
            )
            map_adducts = True
            
        if bool_matrix is not None:
            mask = da.sum(bool_matrix, axis=-1) > 0
            print('Searching Precursors...')
            bool_matrix,mask = dask.compute(bool_matrix,mask)
            print('Decoding hit matrix to indices...')
            mask = db.from_sequence(mask,partition_size=1)
            frag_indexs = mask.map(lambda x: np.argwhere(x).flatten())
            frag_indexs = frag_indexs.compute(scheduler='threads')
        
        ref_embedding = self.mol_embedding_array(embedding_name)
        
        print('Searching MolLib...')
        index,scores = search_tools.search_embeddings(
            query_embedding,
            ref_embedding,
            mask,top_k,
        )
        
        print('Decoding matrix to Smiles...')
        index_db = db.from_sequence(index)
        
        smiles = index_db.map(lambda x: self.SMILES.values[x] if x is not None else None)
        smiles = smiles.compute(scheduler='threads')
        results:Dict[str,np.ndarray] = {
            'index': index,
            'smiles': smiles,
            'score': scores,
        }
        
        if map_adducts is True:
            print('Decoding matrix to Fragments...')
            fragments = search_tools.decode_matrix_to_fragments(
                formulas=self.SMILES,
                adducts=pd.Series(adducts),
                bool_matrix=bool_matrix,
                select_list=index,
            )
            results['formula'] = fragments['formula']
            results['adduct'] = fragments['adduct']
            
        return results
    
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> MolLib:
        iloc = self.format_selection(i_and_key)
        new_lib = MolLib()
        new_lib.molecules = self.molecules.select(iloc)
        new_lib.fragments = self.fragments.select(iloc)
        new_lib.embeddings = self.embeddings.select(iloc)
        new_lib.index = self.index[iloc]
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> MolLib:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> MolLib:
        return base_tools.load_pickle(path)