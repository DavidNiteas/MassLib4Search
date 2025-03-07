from __future__ import annotations
import dask
import dask.bag as db
import dask.array as da
import dask.dataframe as dd
from functools import partial
import numpy as np
import pandas as pd
from . import tools
from rich.progress import track
from rich import print
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

# dev import
import sys
sys.path.append('.')
from MZInferrer.mzinferrer.mz_infer_tools import Fragment,Adduct

class BaseLib(ABC):
    
    @property
    @abstractmethod
    def index(self) -> pd.Index:
        pass
    
    def __len__(self):
        return len(self.index)
    
    def format_i_selection(
        self,
        i: Union[int,slice,Sequence[int],None],
    ) -> List[int]:
        match i:
            case int():
                return [i]
            case slice():
                return range(*i.indices(len(self)))
            case Sequence():
                return i
            case None:
                return []
            case _:
                raise TypeError(f"select index must be int, slice or sequence, not {type(i)}")
    
    def key2i(self, key: Sequence[Hashable]) -> List[int]:
        return list(self.index.get_indexer_for(key))
            
    def format_key_selection(
        self,
        key: Union[Hashable,Sequence[Hashable],None], 
    ) -> List[int]:
        match key:
            case Hashable():
                return self.key2i([key])
            case Sequence():
                return self.key2i(key)
            case None:
                return []
            case _:
                raise TypeError(f"select key must be hashable or sequence, not {type(key)}")
            
    def format_bool_selection(
        self,
        bool_selection: Sequence[bool],
    ) -> List[int]:
        return np.argwhere(bool_selection).flatten().tolist()
    
    def format_selection(
        self,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]],
    ) -> List[int]:
        iloc = []
        match i_and_key:
            case int() | slice() | Sequence() if not isinstance(i_and_key, tuple):
                iloc.extend(self.format_i_selection(i_and_key))
            case tuple():
                if len(i_and_key) >= 1:
                    iloc.extend(self.format_i_selection(i_and_key[0]))
                if len(i_and_key) >= 2:
                    iloc.extend(self.format_key_selection(i_and_key[1]))
                if len(i_and_key) >= 3:
                    iloc.extend(self.format_bool_selection(i_and_key[2]))
        return iloc
    
    def item_select(
        self,
        item:Union[pd.Series,pd.DataFrame,np.ndarray,Any],
        iloc:Sequence[int],
    ) -> Union[pd.Series,pd.DataFrame,np.ndarray,Any]:
        if hasattr(item, "__len__"):
            if len(item) > 0:
                match item:
                    case pd.DataFrame():
                        return item.iloc[iloc]
                    case pd.Series():
                        if len(item) == len(self):
                            return item.iloc[iloc]
                        else:
                            return item.map(lambda x: self.item_select(x, iloc))
                    case np.ndarray():
                        return item[iloc]
                    case BaseLib():
                        return item.select(iloc)
                    case _:
                        if hasattr(item, "__getitem__"):
                            return item[iloc]
                        else:
                            raise TypeError(f"item must be a splitable object, not {type(item)}")
        else:
            raise TypeError(f"item must be a lengthable object, not {type(item)}")
    
    @abstractmethod
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> BaseLib:
        pass
    
    def __getitem__(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> BaseLib:
        return self.select(i_and_key)
    
    def to_bytes(self) -> bytes:
        return tools.to_pickle_bytes(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> BaseLib:
        return tools.from_pickle_bytes(data)
    
    def to_pickle(self, path: str):
        tools.save_pickle(self, path)
    
    @classmethod
    def from_pickle(cls, path: str) -> BaseLib:
        return tools.load_pickle(path)
    
class FragLib(BaseLib):
    
    non_adduct_columns = frozenset(['formula', 'RT', '[M]'])
    defualt_positive_adducts = ['[M]','[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+']
    defualt_negative_adducts = ['[M]', '[M-H]-', '[M+COOH]-', '[M+CH3COO]-']
    
    @staticmethod
    def lazy_init(
        formula: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        npartitions: Optional[int] = None,
    ) -> Dict[
        Union[Literal['formula', 'RT', 'adducts'],str],
        Union[
            db.Bag,
            List[str], # Adducts
            List[Optional[float]], #RT
        ]
    ]:
        if not isinstance(formula, db.Bag):
            formula = db.from_sequence(formula,npartitions=npartitions)
            
        if adducts == 'pos':
            adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in FragLib.defualt_positive_adducts]
        elif adducts == 'neg':
            adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in FragLib.defualt_negative_adducts]
        else:
            adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in adducts]
            
        exact_masses = formula.map(lambda x: Fragment.from_formula_string(x).ExactMass if x is not None else None)
            
        lazy_dict = {"formula": formula, 'adducts': adducts}
        for adduct in adducts:
            predict_mz_func = partial(tools.predict_adduct_mz, adduct)
            lazy_dict[str(adduct)] = exact_masses.map(predict_mz_func)
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
            adduct: Adduct
            adduct_string = str(adduct)
            self.fragments[adduct_string] = computed_lazy_dict[adduct_string]
        if 'RT' in computed_lazy_dict:
            self.fragments['RT'] = computed_lazy_dict['RT']
        self.fragments = pd.DataFrame(self.fragments)

    def __init__(
        self, 
        formula: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if formula is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(formula, RT, adducts, num_workers)
                (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
            self.from_lazy(**computed_lazy_dict)
    
    def add_adducts(
        self,
        adducts: List[str],
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in adducts if adduct_string not in self.Adducts]
        new_fragments:Dict[str,List[float]] = {}
        if '[M]' in self.Fragments.columns:
            exact_masses = db.from_sequence(self.Fragments['[M]'],npartitions=num_workers)
        else:
            formulas = db.from_sequence(self.Formulas,npartitions=num_workers)
            exact_masses = formulas.map(lambda x: Fragment.from_formula_string(x).ExactMass if x is not None else None)
        for adduct in adducts:
            predict_mz_func = partial(tools.predict_adduct_mz, adduct)
            new_fragments[str(adduct)] = exact_masses.map(predict_mz_func)
        (new_fragments,) = dask.compute(new_fragments,scheduler=scheduler,num_workers=num_workers)
        for adduct_string, adduct_mzs in new_fragments.items():
            self.Fragments[adduct_string] = adduct_mzs
    
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
    
    def get_MZs_by_adducts(self, adducts: List[str]) -> pd.DataFrame:
        return self.Fragments[adducts]
    
    @property
    def RTs(self) -> Optional[pd.Series]:
        return self.Fragments.get('RT', None)
    
    @property
    def bad_index(self) -> pd.Index:
        return self.index[self.Formulas.isna()]
        
    def search_to_matrix(
        self,
        query_mzs: Union[np.ndarray,da.Array], # shape: (n_ions,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[np.ndarray] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
    ) -> da.Array: # shape: (n_ions, n_fragments, n_adducts)
        '''
        Search chemical formulas in reference library and return a boolean presence matrix \n
        在参考库中搜索化学式并返回布尔值存在矩阵
        
        Args:
            query_mzs: Array of m/z values to search (待查询的m/z数组)
            adducts: 
                - 'all_adducts': Consider all non-parent adducts (考虑所有非母体加合物)
                - 'no_adducts': Only consider parent ion [M] (仅考虑母体离子[M])
                - Custom list: Specify adducts to consider (自定义加合物列表)
            mz_tolerance: Mass-to-charge tolerance (质荷比容差)
            mz_tolerance_type: 'ppm' (parts-per-million) or 'Da' (Daltons) (容差类型)
            query_RTs: Retention times for queries (optional) (查询保留时间，可选)
            RT_tolerance: Retention time tolerance in minutes (保留时间容差/分钟)
            adduct_co_occurrence_threshold: Minimum number of required adduct co-occurrences (加合物共现的最小数量阈值)
        
        Returns:
            da.Array: Boolean matrix (n_queries, n_formulas, n_adducts) indicating formula presence (三维命中矩阵)
            
        Note:
            This function is a lazy operation, it will not compute anything until you call dask.compute on the result.
            该函数是一个懒惰操作,直到你调用dask.compute来计算结果时才会进行计算。
        '''
        
        if isinstance(adducts, str):
            if adducts == 'all_adducts':
                adducts = self.Adducts
            elif adducts == 'no_adducts':
                adducts = ['[M]']
                
        ref_fragment_table = self.get_MZs_by_adducts(adducts)
        
        results = tools.search_fragments_to_matrix(
            ref_fragment_table.values,query_mzs,
            mz_tolerance,mz_tolerance_type,
            self.RTs,query_RTs,RT_tolerance,
            adduct_co_occurrence_threshold,
        )
        return results
        
    def search(
        self,
        query_mzs: Union[np.ndarray,da.Array], # shape: (n_ions,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Union[np.ndarray,da.Array,None] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        return_matrix: bool = False, # if True, return a matrix of shape (n_queries, n_formulas), dtype: bool, indicating which formulas are present for each query
    ) -> Union[
        List[Dict[str, str]], # List[Dict[formula, adduct]]
        Tuple[List[Dict[str, str]], np.ndarray],
    ]:
        '''
        Perform chemical formula search and return matching results \n
        执行化学式搜索并返回匹配结果
        
        Args:
            query_mzs: Array of m/z values to search (待查询的m/z数组)
            adducts: Adduct selection mode (加合物选择模式)
            mz_tolerance: Mass-to-charge tolerance (质荷比容差)
            mz_tolerance_type: 'ppm' or 'Da' (容差类型)
            query_RTs: Retention times for queries (optional) (查询保留时间，可选)
            RT_tolerance: Retention time tolerance (保留时间容差)
            adduct_co_occurrence_threshold: Minimum adduct co-occurrences (加合物共现阈值)
            return_matrix: Whether to return presence matrix (是否返回存在矩阵)
        
        Returns:
            Union:
                - List[Dict]: For each query, dictionary of {formula: adduct} (匹配结果字典列表)
                - Tuple: (Results list, presence matrix) when return_matrix=True (包含结果和命中矩阵的元组)
        
        Note:
            When using RT filtering, both query_RTs and library RTs must be provided
            使用保留时间筛选时，必须提供查询和库的保留时间数据
        '''
        if adducts == 'all_adducts':
            adducts = self.Adducts
        elif adducts == 'no_adducts':
            adducts = ['[M]']
        ref_fragment_table = self.get_MZs_by_adducts(adducts)
        ref_RTs = self.RTs.values if self.RTs is not None else None
        
        results = tools.search_fragments(
            self.Formulas,ref_fragment_table,query_mzs,
            mz_tolerance,mz_tolerance_type,
            ref_RTs,query_RTs,RT_tolerance,
            adduct_co_occurrence_threshold,
            return_matrix,
        )
        return results
    
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> FragLib:
        iloc = self.format_selection(i_and_key)
        fragments_table = self.item_select(self.Fragments,iloc)
        new_lib = FragLib()
        new_lib.fragments = fragments_table
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> FragLib:
        return tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> FragLib:
        return tools.load_pickle(path)

class MolLib(BaseLib):
    
    @staticmethod
    def lazy_init(
        smiles: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        embeddings: Dict[str, Union[np.ndarray,db.Bag,da.Array]] = {},
        peak_patterns: Union[
            List[
                Dict[
                    Hashable, # fragment name/id in ms2
                    float # fragment m/z in ms2
                ]
            ],
            db.Bag,
            None,
        ] = None,
        loss_patterns: Union[
            List[
                Tuple[
                    Dict[Hashable, float], # {delta name/id in ms2: delta m/z}
                    List[List[Hashable]], # the sequence of loss pathways
                ]
            ],
            db.Bag,
            None,
        ] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['smiles', 'embeddings', 'fragments', 'peak_patterns', 'loss_patterns'],
        Union[
            db.Bag,
            List[str], # Adducts
            List[Optional[float]], #RT
            List[Dict[Hashable, float]], # peak patterns
            List[Tuple[Dict[Hashable, float], List[List[Hashable]]]] # loss patterns
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
            'peak_patterns': peak_patterns,
            'loss_patterns': loss_patterns,
        }
        
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.smiles = pd.Series(computed_lazy_dict['smiles'])
        self.fragments = FragLib(computed_lazy_dict=computed_lazy_dict['fragments'])
        self.embeddings = pd.Series(computed_lazy_dict['embeddings'])
        self.peak_patterns = pd.Series()
        if 'peak_patterns' in computed_lazy_dict:
            if computed_lazy_dict['peak_patterns'] is not None:
                self.peak_patterns = pd.Series(computed_lazy_dict['peak_patterns'])
        self.loss_patterns = pd.Series()
        if 'loss_patterns' in computed_lazy_dict:
            if computed_lazy_dict['loss_patterns'] is not None:
                self.loss_patterns = pd.Series(computed_lazy_dict['loss_patterns'])
    
    def __init__(
        self,
        smiles: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        embeddings: Dict[str, Union[np.ndarray,db.Bag,da.Array]] = {},
        peak_patterns: Union[
            List[
                Dict[
                    Hashable, # fragment name/id in ms2
                    float # fragment m/z in ms2
                ]
            ],
            db.Bag,
            None,
        ] = None,
        loss_patterns: Union[
            List[
                Tuple[
                    Dict[Hashable, float], # {delta name/id in ms2: delta m/z}
                    List[List[Hashable]], # the sequence of loss pathways
                ]
            ],
            db.Bag,
            None,
        ] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if smiles is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(smiles, RT, adducts, embeddings, peak_patterns, loss_patterns, num_workers)
                (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
            self.from_lazy(**computed_lazy_dict)
    
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
    
    @property
    def PeakPatterns(self) -> pd.Series:
        return self.peak_patterns
    
    @property
    def LossPatterns(self) -> pd.Series:
        return self.loss_patterns
    
    def mol_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        return self.embeddings.get(embedding_name, None)
    
    def search_embedding(
        self,
        embedding_name: str,
        query_embedding: Union[np.ndarray,da.Array], # shape: (n_queries, embedding_dim)
        query_mzs: Union[np.ndarray,da.Array,None] = None, # shape: (n_ions,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Union[np.ndarray,da.Array,None] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
    ) -> Dict[
        Literal['index','smiles','score','formula','adduct'],
        np.ndarray, # array of indices, array of SMILES strings, array of scores, array of formulas, array of adducts
    ]:
        
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
        
        ref_embedding = self.mol_embedding(embedding_name)
        if ref_embedding is None:
            raise ValueError(f"No embedding named {embedding_name} found in MolLib")
        
        print('Searching MolLib...')
        index,scores = tools.search_embeddings(
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
            fragments = tools.decode_matrix_to_fragments(
                formulas=self.SMILES,
                adducts=pd.Series(adducts),
                bool_matrix=bool_matrix,
                index_list=index,
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
        new_lib.smiles = self.item_select(self.smiles,iloc)
        new_lib.fragments = self.item_select(self.fragments,iloc)
        new_lib.embeddings = self.item_select(self.embeddings,iloc)
        new_lib.peak_patterns = self.item_select(self.peak_patterns,iloc)
        new_lib.loss_patterns = self.item_select(self.loss_patterns,iloc)
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> MolLib:
        return tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> MolLib:
        return tools.load_pickle(path)

class SpecLib(BaseLib):
    
    @staticmethod
    def lazy_init(
        PI: Union[List[float],db.Bag,None] = None,
        RT: Union[List[float],db.Bag,None] = None,
        mzs: Union[List[List[float]],db.Bag,None] = None,
        intensities: Union[List[List[float]],db.Bag,None] = None,
        spec_embeddings: Dict[str, Union[np.ndarray,da.Array]] = {},
        smiles: Union[List[str],db.Bag,None] = None,
        mol_RT: Union[List[Optional[float]],db.Bag,None] = None,
        mol_adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        mol_embeddings: Dict[str, Union[np.ndarray,da.Array]] = {},
        peak_patterns: Union[
            List[
                Dict[
                    Hashable, # fragment name/id in ms2
                    float # fragment m/z in ms2
                ]
            ],
            db.Bag,
            None,
        ] = None,
        loss_patterns: Union[
            List[
                Tuple[
                    Dict[Hashable, float], # {delta name/id in ms2: delta m/z}
                    List[List[Hashable]], # the sequence of loss pathways
                ]
            ],
            db.Bag,
            None,
        ] = None,
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
        mol_lib_lazy_dict = MolLib.lazy_init(smiles, mol_RT, mol_adducts, mol_embeddings, peak_patterns, loss_patterns, npartitions)
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
        mol_adducts: Union[List[str],Literal['pos','neg']] = 'pos', # list of adducts for each SMILES string
        mol_embeddings: Dict[str, np.ndarray] = {}, # dict of embedding arrays, keyed by embedding name
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not isinstance(computed_lazy_dict,dict):
            lazy_dict = self.lazy_init(PI, RT, mzs, intensities, spec_embeddings, smiles, mol_RT, mol_adducts, mol_embeddings, num_workers)
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
    def FragmentLib(self) -> FragLib:
        return self.MolLib.FragmentLib
    
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
    
    @property
    def PeakPatterns(self) -> pd.Series:
        return self.MolLib.PeakPatterns
    
    @property
    def LossPatterns(self) -> pd.Series:
        return self.MolLib.LossPatterns
    
    def spec_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        return self.spec_embeddings.get(embedding_name, None)
    
    def mol_embedding(self, embedding_name: str) -> Optional[np.ndarray]:
        return self.mols.mol_embedding(embedding_name)
    
    def search_precursors_to_matrix(
        self,
        query_precursor_mzs: Union[da.Array, np.ndarray], # shape: (n_queries,)
        precursor_mz_tolerance: float = 5,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_precursor_RTs: Optional[Union[da.Array, np.ndarray]] = None,
        precursor_RT_tolerance: float = 0.1,
    ) -> da.Array: # shape: (n_queries, n_formulas), dtype: bool
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
        query_precursor_mzs: Union[da.Array, np.ndarray], # shape: (n_queries,)
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_precursor_RTs: Optional[Union[da.Array, np.ndarray]] = None,
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
        query_embedding: Union[np.ndarray,da.Array], # shape: (n_queries, embedding_dim)
        query_precursor_mzs: Union[np.ndarray,da.Array,None] = None, # shape: (n_queries,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        precursor_mz_tolerance: float = 10,
        precursor_mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Union[np.ndarray,da.Array,None] = None,
        RT_tolerance: float = 0.1,
        adduct_co_occurrence_threshold: int = 1, # if a formula has less than this number of adducts, it will be removed from the result
        top_k: int = 5, # number of hits to return for each query
        precursor_search_type: Literal['mol', 'spec'] = 'mol',
    ) -> Dict[
        Literal['index','smiles','score','formula','adduct'],
        np.ndarray, # array of indices, array of SMILES strings, array of scores, array of formulas, array of adducts
    ]:
        
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
            
        if query_precursor_mzs is not None and len(self.precursors) > 0 and precursor_search_type =='spec':
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
        
        ref_embedding = self.spec_embedding(embedding_name)
        if ref_embedding is None:
            raise ValueError(f"No embedding named {embedding_name} found in SpecLib.")
        
        print('Searching SpecLib...')
        index,scores = tools.search_embeddings(
            query_embedding,
            ref_embedding,
            frag_indexs,top_k,
        )

        print('Decoding matrix to Smiles...')
        index_db = db.from_sequence(index)
        
        smiles = index_db.map(lambda x: self.mol_smiles.values[x] if x is not None else None)
        smiles = smiles.compute(scheduler='threads')
        results:Dict[str,np.ndarray] = {
            'index': index,
            'smiles': smiles,
            'score': scores,
        }
        
        if map_adducts is True:
            print('Decoding matrix to Fragments...')
            fragments = tools.decode_matrix_to_fragments(
                formulas=self.mol_formulas,
                adducts=pd.Series(adducts),
                bool_matrix=bool_matrix,
                index_list=index,
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
        new_lib.precursors = self.item_select(self.precursors,iloc)
        new_lib.peaks = self.item_select(self.peaks,iloc)
        new_lib.spec_embeddings = self.item_select(self.spec_embeddings,iloc)
        new_lib.mols = self.item_select(self.mols,iloc)
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> SpecLib:
        return tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> SpecLib:
        return tools.load_pickle(path)