from __future__ import annotations

from .lib_utils import io
from .search_utils import search_tools
from .base_lib import BaseLib,CriticalDataMissingError
import dask
import dask.bag as db
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
    
    row_major_schemas_tags = {
        'fragments': CriticalDataMissingError("Fragments data is missing, which is required for FragLib"),
        'metadatas': None,
        "index": None,
    }
    
    non_adduct_columns = frozenset(['formula', 'RT', '[M]'])
    defualt_positive_adducts = ['[M]','[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+']
    defualt_negative_adducts = ['[M]', '[M-H]-', '[M+COOH]-', '[M+CH3COO]-']
    
    @classmethod
    def lazy_init(
        cls,
        formula: Union[List[str],db.Bag],
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        metadatas: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Union[Literal['index', 'formula', 'RT', 'adducts'],str],
        Union[
            db.Bag,
            List[str], # Adducts
            List[Optional[float]], # RT
            pd.Index,Sequence[Hashable], # index
            pd.DataFrame, # metadatas
            None,
        ]
    ]:
        if name is None:
            name = cls.get_default_name()
        
        if not isinstance(formula, db.Bag):
            formula = db.from_sequence(formula,npartitions=npartitions)
            
        if adducts == 'pos':
            adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in FragLib.defualt_positive_adducts]
        elif adducts == 'neg':
            adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in FragLib.defualt_negative_adducts]
        else:
            adducts:List[Adduct] = [Adduct.from_adduct_string(adduct_string) for adduct_string in adducts]
            
        exact_masses = formula.map(lambda x: Fragment.from_formula_string(x).ExactMass if x is not None else None)
        
        lazy_dict = {
            "index": index, 
            "formula": formula, 
            'adducts': adducts, 
            'metadatas': metadatas,
            'name': name,
        }
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
        self.metadatas = computed_lazy_dict.get('metadatas', None)
        self.name = computed_lazy_dict['name']

    def __init__(
        self, 
        formula: Union[List[str],db.Bag,None] = None,
        RT: Optional[List[Optional[float],db.Bag]] = None,
        adducts: Union[List[str],Literal['pos','neg']] = 'pos',
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        metadatas: Optional[pd.DataFrame] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        # 确保始终初始化name属性
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.get_default_name()
            
        if formula is not None or computed_lazy_dict is not None:
            if not isinstance(computed_lazy_dict, dict):
                lazy_dict = type(self).lazy_init(formula, RT, adducts, index, metadatas, num_workers)
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
    def Metadatas(self) -> Optional[pd.DataFrame]:
        return self.metadatas
    
    @property
    def bad_index(self) -> pd.Index:
        return self.index[self.Formulas.isna()]
        
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
        qry_chunks: int = 5120,
        ref_chunks: int = 5120,
        I_F_D_matrix_chunks: Tuple[int, int, int] = (10240, 10240, 5)
    ) -> pd.DataFrame: # columns: db_index, formula, adduct

        if adducts == 'all_adducts':
            adducts = self.Adducts
        elif adducts == 'no_adducts':
            adducts = ['[M]']
        ref_fragment_table = self.get_MZs_by_adducts(adducts)
        ref_RTs = self.RTs.values if self.RTs is not None else None
        
        results = search_tools.search_fragments(
            qry_ions=query_mzs,
            ref_fragment_mzs=ref_fragment_table,
            ref_fragment_formulas=self.Formulas,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            ref_RTs=ref_RTs,
            query_RTs=query_RTs,
            RT_tolerance=RT_tolerance,
            adduct_co_occurrence_threshold=adduct_co_occurrence_threshold,
            batch_size=batch_size,
            qry_chunks=qry_chunks,
            ref_chunks=ref_chunks,
            I_F_D_matrix_chunks=I_F_D_matrix_chunks,
        )
        
        return results
    
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> FragLib:
        if self.is_empty:
            new_lib = self.__class__()
            if hasattr(self, 'name'):
                new_lib.name = self.name
            else:
                new_lib.name = self.__class__.get_default_name()
            return new_lib
        iloc = self.format_selection(i_and_key)
        new_lib = FragLib()
        new_lib.fragments = self.item_select(self.Fragments,iloc)
        new_lib.metadatas = self.item_select(self.Metadatas,iloc)
        new_lib.index = self.Index[iloc]
        if hasattr(self, 'name'):
            new_lib.name = self.name
        else:
            new_lib.name = self.__class__.get_default_name()
        return new_lib
    
    @classmethod
    def from_bytes(cls, data: bytes) -> FragLib:
        return io.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> FragLib:
        return io.load_pickle(path)