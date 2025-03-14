from __future__ import annotations
import numpy as np
import pandas as pd
import dask
import dask.bag as db
import dask.array as da
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from . import base_tools
from abc import ABC, abstractmethod
from numpy.typing import NDArray,ArrayLike
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

class LibBase(ABC):
    
    @staticmethod
    def get_index(computed_lazy_dict:dict) -> Optional[pd.Index]:
        if "index" in computed_lazy_dict:
            if computed_lazy_dict['index'] is not None:
                return pd.Index(computed_lazy_dict['index'])
        lens = []
        for item in computed_lazy_dict.values():
            if hasattr(item,"__len__"):
                lens.append(len(item))
        if len(lens) > 0:
            return pd.RangeIndex(0,max(lens))
        else:
            return None
        
    @staticmethod
    def init_data_item(computed_lazy_dict:dict,index:pd.Index,name:str) -> pd.Series:
        item = computed_lazy_dict.get(name)
        match item:
            case pd.Series():
                return item.set_index(index)
            case None:
                return None
            case _:
                assert len(item) == len(index), f"the len of {name} is {len(item)} but the len of index is {len(index)}"
                if isinstance(item, np.ndarray):
                    if len(item.shape) > 1:
                        item = [arr for arr in item]
                return pd.Series(item,index=index)
    
    @property
    @abstractmethod
    def Index(self) -> Optional[pd.Index]:
        pass
    
    def __len__(self) -> int:
        if self.Index is None:
            return 0
        return len(self.Index)
    
    def format_i_selection(
        self,
        i: Union[int, slice, Sequence[int], None],
    ) -> List[int]:
        if isinstance(i, int):
            return [i]
        elif isinstance(i, slice):
            return range(*i.indices(len(self)))
        elif isinstance(i, Sequence):
            return i
        elif i is None:
            return []
        else:
            raise TypeError(f"select index must be int, slice or sequence, not {type(i)}")
    
    def key2i(self, key: Sequence[Hashable]) -> List[int]:
        return list(self.Index.get_indexer_for(key))
            
    def format_key_selection(
        self,
        key: Union[Hashable,Sequence[Hashable],None], 
    ) -> List[int]:
        if isinstance(key, Hashable):
            return self.key2i([key])
        elif isinstance(key, Sequence):
            return self.key2i(key)
        elif key is None:
            return []
        else:
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
        if isinstance(i_and_key, tuple):
            if len(i_and_key) >= 1:
                iloc.extend(self.format_i_selection(i_and_key[0]))
            if len(i_and_key) >= 2:
                iloc.extend(self.format_key_selection(i_and_key[1]))
            if len(i_and_key) >= 3:
                iloc.extend(self.format_bool_selection(i_and_key[2]))
        else:
            iloc.extend(self.format_i_selection(i_and_key))
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
                    case LibBase():
                        return item.select(iloc)
                    case _:
                        if hasattr(item, "__getitem__"):
                            return item[iloc]
                        else:
                            return item
            else:
                return item
        else:
            return item
    
    @abstractmethod
    def select(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> LibBase:
        pass
    
    def __getitem__(
        self, 
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> LibBase:
        return self.select(i_and_key)
    
    def to_bytes(self) -> bytes:
        return base_tools.to_pickle_bytes(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> LibBase:
        return base_tools.from_pickle_bytes(data)
    
    def to_pickle(self, path: str):
        base_tools.save_pickle(self, path)
    
    @classmethod
    def from_pickle(cls, path: str) -> LibBase:
        return base_tools.load_pickle(path)
    
    @property
    def is_empty(self) -> bool:
        return len(self) == 0
    
class BaseLibBlock(LibBase):
    
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        pass
    
    @classmethod
    @abstractmethod
    def from_dataframe(dataframe: pd.DataFrame) -> BaseLibBlock:
        pass
    
class Spectrums(BaseLibBlock):
    
    @staticmethod
    def lazy_init(
        PI: Union[ArrayLike[float],db.Bag,None] = None,
        RT: Union[ArrayLike[float],db.Bag,None] = None,
        mzs: Union[Sequence[ArrayLike[float]],db.Bag,None] = None,
        intensities: Union[Sequence[ArrayLike[float]],db.Bag,None] = None,
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['PI', 'RT','mzs', 'intensities', 'index'],
        Union[
            db.Bag,
            List[float], # PI, RT
            List[List[float]], # mzs, intensities
            pd.Index,Sequence[Hashable], # index
            None,
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
        return {
            'PI': PI,
            'RT': RT,
            'mzs': mzs,
            'intensities': intensities,
            'index': index,
        }
    
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.index = self.get_index(computed_lazy_dict)
        self.precursor_mzs = self.init_data_item(computed_lazy_dict,self.index,"PI")
        self.precursor_rts = self.init_data_item(computed_lazy_dict,self.index,"RT")
        self.spectrum_mzs = self.init_data_item(computed_lazy_dict,self.index,"mzs")
        self.spectrum_intens = self.init_data_item(computed_lazy_dict,self.index,"intensities")
    
    def __init__(
        self,
        PI: Union[ArrayLike[float],db.Bag,None] = None,
        RT: Union[ArrayLike[float],db.Bag,None] = None,
        mzs: Union[Sequence[ArrayLike[float]],db.Bag,None] = None,
        intensities: Union[Sequence[ArrayLike[float]],db.Bag,None] = None,
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not isinstance(computed_lazy_dict,dict):
            lazy_dict = self.lazy_init(PI, RT, mzs, intensities, index, npartitions=num_workers)
            (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
        self.from_lazy(**computed_lazy_dict)
    
    @property
    def Index(self) -> Optional[pd.Index]:
        return self.index
    
    @property
    def PIs(self) -> Optional[pd.Series]:
        return self.precursor_mzs
    
    @property
    def RTs(self) -> Optional[pd.Series]:
        return self.precursor_rts
    
    @property
    def MZs(self) -> Optional[pd.Series]:
        return self.spectrum_mzs
    
    @property
    def Intens(self) -> Optional[pd.Series]:
        return self.spectrum_intens
    
    def select(
        self,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> Spectrums:
        if self.is_empty:
            return self.__class__()
        iloc = self.format_selection(i_and_key)
        new_spectrums = Spectrums()
        new_spectrums.precursor_mzs = self.item_select(self.precursor_mzs, iloc)
        new_spectrums.precursor_rts = self.item_select(self.precursor_rts, iloc)
        new_spectrums.spectrum_mzs = self.item_select(self.spectrum_mzs, iloc)
        new_spectrums.spectrum_intens = self.item_select(self.spectrum_intens, iloc)
        new_spectrums.index = self.index[iloc]
        return new_spectrums
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Spectrums:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> Spectrums:
        return base_tools.load_pickle(path)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'PI': self.precursor_mzs,
            'RT': self.precursor_rts,
            'mzs': self.spectrum_mzs,
            'intensities': self.spectrum_intens,
        },index=self.index)
        
    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Spectrums:
        spectrums = Spectrums()
        if not dataframe.empty:
            if 'PI' in dataframe.columns:
                spectrums.precursor_mzs = dataframe['PI']
            if 'RT' in dataframe.columns:
                spectrums.precursor_rts = dataframe['RT']
            if'mzs' in dataframe.columns:
                spectrums.spectrum_mzs = dataframe['mzs']
            if 'intensities' in dataframe.columns:
                spectrums.spectrum_intens = dataframe['intensities']
            spectrums.index = dataframe.index
        return spectrums
        
class Molecules(BaseLibBlock):
    
    @staticmethod
    def lazy_init(
        smiles: Union[List[str],db.Bag,None] = None,
        inchi: Union[List[str],db.Bag,None] = None,
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        Literal['smiles', 'inchi', 'index'],
        Union[
            db.Bag,
            List[str], # smiles, inchi
            pd.Index,Sequence[Hashable], # index
            None,
        ]
    ]:
        if smiles is None and inchi is not None:
            smiles = db.from_sequence(inchi,npartitions=npartitions).map(lambda x: Chem.MolToSmiles(Chem.MolFromInchi(x)))
        return {
            'smiles': smiles,
            'inchi': inchi,
            'index': index,
        }
    
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.index = self.get_index(computed_lazy_dict)
        self.smiles = self.init_data_item(computed_lazy_dict,self.index,"smiles")
        self.inchi = self.init_data_item(computed_lazy_dict,self.index,"inchi")
    
    def __init__(
        self,
        smiles: Union[List[str],db.Bag,None] = None,
        inchi: Union[List[str],db.Bag,None] = None,
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        scheduler: Optional[str] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not isinstance(computed_lazy_dict,dict):
            lazy_dict = self.lazy_init(smiles, inchi, index, npartitions=num_workers)
            (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
        self.from_lazy(**computed_lazy_dict)
    
    @property
    def Index(self) -> Optional[pd.Index]:
        return self.index
    
    @property
    def SMILES(self) -> pd.Series:
        return self.smiles
    
    @property
    def InChi(self) -> Optional[pd.Series]:
        return self.inchi
    
    def select(
        self,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> Molecules:
        if self.is_empty:
            return self.__class__()
        iloc = self.format_selection(i_and_key)
        new_molecules = Molecules()
        new_molecules.smiles = self.item_select(self.smiles, iloc)
        new_molecules.inchi = self.item_select(self.inchi, iloc)
        new_molecules.index = self.index[iloc]
        return new_molecules
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Molecules:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> Molecules:
        return base_tools.load_pickle(path)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'smiles': self.smiles,
            'inchi': self.inchi,
        },index=self.index)
        
    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Molecules:
        molecules = Molecules()
        if not dataframe.empty:
            if'smiles' in dataframe.columns:
                molecules.smiles = dataframe['smiles']
            if 'inchi' in dataframe.columns:
                molecules.inchi = dataframe['inchi']
            molecules.index = dataframe.index
        return molecules
    
class Embeddings(BaseLibBlock):
    
    @staticmethod
    def lazy_init(
        embeddings: Dict[str, Union[NDArray[np.float32],db.Bag,da.Array]] = {},
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        npartitions: Optional[int] = None,
    ) -> Dict[
        str,
        Union[
            db.Bag, da.Array, NDArray[np.float32], # embedding array
            pd.Index,Sequence[Hashable], # index
            None,
        ]
    ]:
        return {
            'index': index,
            **embeddings,
        }
    
    def from_lazy(
        self,
        **computed_lazy_dict
    ) -> None:
        self.index = self.get_index(computed_lazy_dict)
        for name in computed_lazy_dict:
            if name != 'index':
                setattr(self, name, self.init_data_item(computed_lazy_dict,self.index,name))
    
    def __init__(
        self,
        embeddings: Dict[str, Union[NDArray[np.float32],db.Bag,da.Array]] = {},
        scheduler: Optional[str] = None,
        index:Union[pd.Index,Sequence[Hashable],None] = None,
        num_workers: Optional[int] = None,
        computed_lazy_dict: Optional[dict] = None,
    ):
        if not isinstance(computed_lazy_dict,dict):
            lazy_dict = self.lazy_init(embeddings, index, npartitions=num_workers)
            (computed_lazy_dict,) = dask.compute(lazy_dict,scheduler=scheduler,num_workers=num_workers)
        self.from_lazy(**computed_lazy_dict)
    
    @property
    def Index(self) -> Optional[pd.Index]:
        return self.index
    
    def select(
        self,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]]]
    ) -> Embeddings:
        if self.is_empty:
            return self.__class__()
        iloc = self.format_selection(i_and_key)
        new_embeddings = Embeddings()
        for name in self.__dict__:
            if name != 'index':
                new_embeddings.__dict__[name] = self.item_select(self.__dict__[name], iloc)
        new_embeddings.index = self.index[iloc]
        return new_embeddings
    
    def get_embedding(
        self,
        embedding_name: str,
        i_and_key: Union[int,slice,Sequence,Tuple[Union[int,Sequence[int]],Union[Hashable,Sequence[Hashable]],Sequence[bool]],None] = None,
    ) -> NDArray[np.float32]:
        emb = getattr(self, embedding_name)
        if i_and_key is not None:
            emb = self.item_select(emb, self.format_selection(i_and_key))
        return emb
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Embeddings:
        return base_tools.from_pickle_bytes(data)
    
    @classmethod
    def from_pickle(cls, path: str) -> Embeddings:
        return base_tools.load_pickle(path)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            name: self.__dict__[name] for name in self.__dict__ if name != 'index'
        },index=self.index)
        
    @classmethod
    def from_dataframe(cls, dataframe: Optional[pd.DataFrame]) -> Embeddings:
        embeddings = Embeddings()
        if dataframe is not None:
            if not dataframe.empty:
                for name in dataframe.columns:
                    embeddings.__dict__[name] = dataframe[name]
                embeddings.index = dataframe.index
        return embeddings
    
class BaseLib(LibBase):
    
    @abstractmethod
    def to_dataframes(self) -> Dict[str,pd.DataFrame]:
        pass
    
    @classmethod
    @abstractmethod
    def from_dataframes(cls, dataframes: Dict[str,pd.DataFrame]):
        pass