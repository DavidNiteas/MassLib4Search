from __future__ import annotations
import numpy as np
import pandas as pd
from . import tools
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

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