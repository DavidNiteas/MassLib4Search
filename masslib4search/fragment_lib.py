from __future__ import annotations
from . import search_tools,base_tools
from .mass_lib_utils import BaseLib
import dask
import dask.bag as db
import dask.array as da
import numpy as np
import pandas as pd
from functools import partial
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

# dev import
import sys
sys.path.append('.')
from MZInferrer.mzinferrer.mz_infer_tools import Fragment,Adduct

def predict_adduct_mz(adduct: Adduct, mass: Union[float, None]) -> Union[float, None]:
    if mass is None:
        return None
    return adduct.predict_mz(mass)

class FragLib(BaseLib):
    
    non_adduct_columns = frozenset(['formula', 'RT', '[M]'])
    defualt_positive_adducts = ['[M]','[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+']
    defualt_negative_adducts = ['[M]', '[M-H]-', '[M+COOH]-', '[M+CH3COO]-']
    
    @staticmethod
    def lazy_init(
        formula: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Union[Literal['index', 'formula', 'RT', 'adducts'],str],
        Union[
            db.Bag,
            List[str], # Adducts
            List[Optional[float]], #RT
            pd.Index,Sequence[Hashable],
            None,
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
            
        lazy_dict = {"index": index, "formula": formula, 'adducts': adducts}
        for adduct in adducts:
            predict_mz_func = partial(predict_adduct_mz, adduct)
            lazy_dict[str(adduct)] = exact_masses.map(predict_mz_func)
        if RT is not None:
            lazy_dict['RT'] = RT
        return lazy_dict
    
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.index = self.get_index(computed_lazy_dict)
        self.fragments = {
            "formula": computed_lazy_dict["formula"],
        }
        for adduct in computed_lazy_dict['adducts']:
            adduct: Adduct
            adduct_string = str(adduct)
            self.fragments[adduct_string] = computed_lazy_dict[adduct_string]
        if 'RT' in computed_lazy_dict:
            self.fragments['RT'] = computed_lazy_dict['RT']
        self.fragments = pd.DataFrame(self.fragments,index=self.index)

    def __init__(
        self, 
        formula: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if formula is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = self.lazy_init(formula, RT, adducts, index, num_workers)
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
            predict_mz_func = partial(predict_adduct_mz, adduct)
            new_fragments[str(adduct)] = exact_masses.map(predict_mz_func)
        (new_fragments,) = dask.compute(new_fragments,scheduler=scheduler,num_workers=num_workers)
        for adduct_string, adduct_mzs in new_fragments.items():
            self.Fragments[adduct_string] = adduct_mzs
    
    @property
    def Index(self):
        return self.index
    
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
        query_mzs: Union[NDArray[np.float_],da.Array], # shape: (n_ions,)
        adducts: Union[List[str],Literal['all_adducts','no_adducts']] = 'all_adducts', # if 'all_adducts', all adducts (without [M]) will be considered, if 'no_adducts', only the [M] will be considered
        mz_tolerance: float = 3,
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',
        query_RTs: Optional[NDArray[np.float_]] = None,
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
        
        results = search_tools.search_fragments_to_matrix(
            ref_fragment_table.values,query_mzs,
            mz_tolerance,mz_tolerance_type,
            self.RTs,query_RTs,RT_tolerance,
            adduct_co_occurrence_threshold,
        )
        return results
        
    def search(
        self,
        query_mzs: Union[NDArray[np.float_], da.Array],  # 待查询的m/z数组，形状为 (n_ions,)
        adducts: Union[List[str], Literal['all_adducts', 'no_adducts']] = 'all_adducts',  # 加合物选择模式，'all_adducts' 表示考虑所有加合物（不包括 [M]），'no_adducts' 表示只考虑 [M]
        mz_tolerance: float = 3,  # 质荷比容差
        mz_tolerance_type: Literal['ppm', 'Da'] = 'ppm',  # 容差类型，'ppm' 或 'Da'
        query_RTs: Union[NDArray[np.float_], da.Array, None] = None,  # 查询保留时间数组（可选）
        RT_tolerance: float = 0.1,  # 保留时间容差
        adduct_co_occurrence_threshold: int = 1,  # 加合物共现阈值，如果某个分子式的加合物数量少于该阈值，则从结果中移除
        return_matrix: bool = False,  # 是否返回存在的布尔矩阵，形状为 (n_queries, n_formulas)，指示每个查询对应的分子式存在情况
    ) -> Dict[
        Literal['formula', 'adduct', 'db_index'],
        List[NDArray[np.bool_]],  # 列表长度为 n_ions，每个元素的形状为 (tag_formula_num,)
    ]:
        '''
        执行化学式搜索并返回匹配结果

        参数:
            query_mzs: 待查询的m/z数组，形状为 (n_ions,)
            adducts: 加合物选择模式，可以是加合物名称的列表，或字符串 'all_adducts' 表示考虑所有加合物，'no_adducts' 表示只考虑 [M]
            mz_tolerance: 质荷比容差
            mz_tolerance_type: 容差类型，'ppm' 或 'Da'
            query_RTs: 查询保留时间数组（可选），如果提供，则必须同时提供库的保留时间数据
            RT_tolerance: 保留时间容差
            adduct_co_occurrence_threshold: 加合物共现阈值，如果某个分子式的加合物数量少于该阈值，则从结果中移除
            return_matrix: 是否返回存在的布尔矩阵，指示每个查询对应的分子式存在情况
        
        返回:
            Dict 或 Tuple:
                - 如果 return_matrix 为 False，返回匹配结果字典列表，每个字典包含键为命中的分子式，值为加合类型
                - 如果 return_matrix 为 True，返回一个包含结果字典列表和布尔矩阵的元组，布尔矩阵指示每个查询对应的分子式存在情况
        
        注意:
            当使用保留时间筛选时，必须同时提供查询和库的保留时间数据
        '''
        if adducts == 'all_adducts':
            adducts = self.Adducts
        elif adducts == 'no_adducts':
            adducts = ['[M]']
        ref_fragment_table = self.get_MZs_by_adducts(adducts)
        ref_RTs = self.RTs.values if self.RTs is not None else None
        
        results = search_tools.search_fragments(
            self.Formulas, ref_fragment_table, query_mzs,
            mz_tolerance, mz_tolerance_type,
            ref_RTs, query_RTs, RT_tolerance,
            adduct_co_occurrence_threshold,
            return_matrix,
        )
        return results
    
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> FragLib:
        iloc = self.format_selection(i_and_key)
        new_lib = FragLib()
        new_lib.fragments = self.item_select(self.Fragments,iloc)
        new_lib.index = self.Index[iloc]
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> FragLib:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> FragLib:
        return base_tools.load_pickle(path)