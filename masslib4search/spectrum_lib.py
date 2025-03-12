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
        if not isinstance(computed_lazy_dict,dict):
            lazy_dict = self.lazy_init(PI, RT, mzs, intensities, spec_embeddings, smiles, mol_RT, mol_adducts, mol_embeddings, index, num_workers)
            (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
        self.from_lazy(**computed_lazy_dict)
        
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
    
    def spec_embedding_array(
        self, 
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> NDArray[np.float32]:
        return self.spec_embeddings.get_embedding_array(embedding_name, i_and_key)
    
    def mol_embedding_array(
        self,
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> NDArray[np.float32]:
        return self.MolLib.mol_embedding_array(embedding_name, i_and_key)
    
    def search_precursors_to_matrix(
        self,
        query_precursor_mzs: Union[da.Array, NDArray[np.float_]], # shape: (n_queries,)
        precursor_mz_tolerance: float = 5,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_precursor_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,
        precursor_RT_tolerance: float = 0.1,
    ) -> da.Array: # shape: (n_queries, n_formulas), dtype: bool
        assert not isinstance(self.PIs,None), "If you want to search precursors, you need to provide PI values when you create the SpecLib."
        results = search_tools.search_precursors_to_matrix(
            query_precursor_mzs,
            self.PIs.values,
            precursor_mz_tolerance,
            precursor_mz_tolerance_type,
            query_precursor_RTs,
            self.RTs.values,
            precursor_RT_tolerance,
        )
        return results
    
    def search_precursors(
        self,
        query_precursor_mzs: Union[da.Array, NDArray[np.float_]], # shape: (n_queries,)
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_precursor_RTs: Optional[Union[da.Array, NDArray[np.float_]]] = None,
        precursor_RT_tolerance: float = 0.1,
        return_matrix: bool = False, # if True, return a matrix of shape (n_queries, n_formulas), dtype: bool, indicating which formulas are present for each query
    ) -> Union[
        List[List[int]], # List[List[ref_precursor_index]]
        Tuple[List[List[int]], NDArray[np.bool_]],
    ]:
        assert not isinstance(self.PIs,None), "If you want to search precursors, you need to provide PI values when you create the SpecLib."
        results = search_tools.search_precursors(
            query_precursor_mzs,
            self.PIs.values,
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
        query_embedding: Union[NDArray[np.float32],da.Array], # shape: (n_queries, embedding_dim)
        query_precursor_mzs: Union[NDArray[np.float_],da.Array,None] = None, # shape: (n_queries,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Union[NDArray[np.float_],da.Array,None] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
        precursor_search_type: Literal['mol', 'spec'] = 'mol',
    ) -> Dict[
        Literal['index','smiles','score','formula','adduct'],
        Union[
            List[NDArray[np.int_]], # index of hits
            List[NDArray[np.str_]], # smiles/formula/adduct of hits
            List[NDArray[np.float_]], # scores of hits
        ]
    ]:
        """
        在谱库中基于嵌入向量进行搜索，返回与查询嵌入向量最匹配的结果。

        参数:
        - embedding_name: str, 嵌入向量的名称。
        - query_embedding: Union[NDArray[np.float32], da.Array], 查询嵌入向量，形状为 (n_queries, embedding_dim)。
        - query_precursor_mzs: Union[NDArray[np.float_], da.Array, None], 查询前体离子的 m/z 值，形状为 (n_queries,)。
        - adducts: Union[List[str], Literal['all_adducts', 'no_adducts']], 加合物列表或特定选项，默认为 'all_adducts'。
        - precursor_mz_tolerance: float, 前体离子 m/z 的容差，默认为 10。
        - precursor_mz_tolerance_type: Literal['ppm', 'Da'], 前体离子 m/z 容差的类型，默认为 'ppm'。
        - query_RTs: Union[NDArray[np.float_], da.Array, None], 查询保留时间，形状为 (n_queries,)。
        - RT_tolerance: float, 保留时间的容差，默认为 0.1。
        - adduct_co_occurrence_threshold: int, 加合物共现阈值，默认为 1。
        - top_k: int, 每个查询返回的命中数量，默认为 5。
        - precursor_search_type: Literal['mol', 'spec'], 前体离子搜索类型，默认为 'mol'。

        返回:
        - Dict[
            Literal['index', 'smiles', 'score', 'formula', 'adduct'],
            Union[
                List[NDArray[np.int_]],  # 命中结果的索引
                List[NDArray[np.str_]],  # 命中结果的 SMILES 字符串、分子式或加合物
                List[NDArray[np.float_]],  # 命中结果的分数
            ]
        ]: 返回一个字典，包含命中结果的索引、SMILES 字符串、分子式、加合物和分数。
        """
        
        bool_matrix = None
        mask = None
        map_adducts = False
        
        if query_precursor_mzs is not None and len(self.MolLib) > 0 and precursor_search_type == 'mol':
            
            if isinstance(adducts, str):
                if adducts == 'all_adducts':
                    adducts = self.MolLib.Adducts
                elif adducts == 'no_adducts':
                    adducts = ['[M]']
            
            bool_matrix = self.FragmentLib.search_to_matrix(
                query_precursor_mzs,adducts,
                precursor_mz_tolerance,precursor_mz_tolerance_type,
                query_RTs,RT_tolerance,
                adduct_co_occurrence_threshold,
            )
            map_adducts = True
            
        if query_precursor_mzs is not None and len(self.spectrums) > 0 and precursor_search_type =='spec':
            bool_matrix = self.search_precursors_to_matrix(
                query_precursor_mzs,
                precursor_mz_tolerance,
                precursor_mz_tolerance_type,
            )
            
        if bool_matrix is not None:
            mask = da.sum(bool_matrix, axis=-1) > 0
            print('Searching Precursors...')
            bool_matrix,mask = dask.compute(bool_matrix,mask)
            print('Decoding hit matrix to indices...')
            mask = db.from_sequence(mask,partition_size=1)
            frag_indexs = mask.map(lambda x: np.argwhere(x).flatten())
            frag_indexs = frag_indexs.compute(scheduler='threads')
        
        ref_embedding = self.spec_embedding_array(embedding_name)
        
        print('Searching SpecLib...')
        index,scores = search_tools.search_embeddings(
            query_embedding,
            ref_embedding,
            frag_indexs,top_k,
        )

        print('Decoding matrix to Smiles...')
        index_db = db.from_sequence(index)
        
        smiles = index_db.map(lambda x: self.MolSMILES.values[x] if x is not None else None)
        smiles = smiles.compute(scheduler='threads')
        results:Dict[str,np.ndarray] = {
            'index': index,
            'smiles': smiles,
            'score': scores,
        }
        
        if map_adducts is True:
            print('Decoding matrix to Fragments...')
            fragments = search_tools.decode_matrix_to_fragments(
                formulas=self.MolFormulas,
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
    ) -> SpecLib:
        iloc = self.format_selection(i_and_key)
        new_lib = SpecLib()
        new_lib.spectrums = self.spectrums.select(iloc)
        new_lib.spec_embeddings = self.spec_embeddings.select(iloc)
        new_lib.mol_lib = self.mol_lib.select(iloc)
        new_lib.index = self.index[iloc]
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SpecLib:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> SpecLib:
        return base_tools.load_pickle(path)