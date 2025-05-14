from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import TypeVar, Type, Dict, Tuple, Any, Optional, List, Sequence, ClassVar

data_entity_type = TypeVar('data_entity_type', bound='SearchDataEntity')
config_entity_type = TypeVar('config_entity_type', bound='SearchConfigEntity')
results_entity_type = TypeVar('results_entity_type', bound='SearchResultsEntity')

class SearchDataEntity(BaseModel, ABC):
    
    # pydantic 配置
    model_config = ConfigDict(extra='forbid', slots=True, arbitrary_types_allowed=True)
    
    @abstractmethod
    def get_inputs(self) -> Tuple[
        Tuple[Any, ...],  # args
        Dict[str, Any],  # kwargs
    ]:
        '''从搜索数据实体中获取搜索函数输入。'''
        pass
    
    @classmethod
    @abstractmethod
    def from_raw_data(cls: Type[data_entity_type], *args, **kwargs) -> data_entity_type:
        '''将原始搜索数据转换为搜索数据实体。'''
        pass
    
class SearchConfigEntity(BaseModel, ABC):
    
    # pydantic 配置
    model_config = dict(extra='forbid', slots=True, arbitrary_types_allowed=True)
    
    @abstractmethod
    def get_inputs(self) -> Tuple[
        Tuple[Any, ...],  # args
        Dict[str, Any],  # kwargs
    ]:
        '''从搜索配置中获取搜索函数 kwargs。'''
        pass
    
class SearchResultsEntity(BaseModel, ABC):
    
    # pydantic 配置
    model_config = dict(extra='forbid', slots=True, arbitrary_types_allowed=True)
    
    @classmethod
    @abstractmethod
    def from_raw_results(
        cls: Type[results_entity_type], 
        raw_results, 
        data: SearchDataEntity,
    ) -> results_entity_type:
        '''将原始搜索结果转换为搜索结果实体。'''
        pass

class Searcher(BaseModel, ABC):
    
    # pydantic 配置
    model_config = ConfigDict(extra='forbid', slots=True, arbitrary_types_allowed=True)
    
    # 类变量
    input_type: ClassVar[Type[SearchDataEntity]] = SearchDataEntity
    results_type: ClassVar[Type[SearchResultsEntity]] = SearchResultsEntity
    
    # 实例变量
    config: SearchConfigEntity
    
    def check_data(self, data: SearchDataEntity):
        '''检查数据是否适用于搜索。'''
        if not isinstance(data, self.input_type):
            raise TypeError(f"数据必须是类型 {self.input_type.__name__}")
        
    def merge_inputs(
        self,
        data_args: Tuple[Any, ...],
        data_kwargs: Dict[str, Any],
        config_args: Tuple[Any, ...],
        config_kwargs: Dict[str, Any],
    ) -> Tuple[
        Tuple[Any, ...],  # args
        Dict[str, Any],  # kwargs
    ]:
        args = []
        for data, config in zip(data_args, config_args):
            args.append(data if data is not None else config)
        kwargs = {**data_kwargs, **config_kwargs}
        return tuple(args), kwargs
    
    @abstractmethod
    def search_method(self, *args, **kwargs):
        '''
        需要实现的搜索函数。
        该函数应以搜索数据和配置作为输入。
        该函数应返回原始搜索结果，这些结果将由 SearchResultsEntity.from_raw_results() 方法转换为搜索结果实体。
        '''
        pass
    
    def run(self, data: SearchDataEntity) -> SearchResultsEntity:
        '''使用给定的数据和配置运行搜索函数。'''
        self.check_data(data)
        data_args, data_kwargs = data.get_inputs()
        config_args, config_kwargs = self.config.get_inputs()
        args, kwargs = self.merge_inputs(data_args, data_kwargs, config_args, config_kwargs)
        raw_results = self.search_method(*args, **kwargs)
        results = self.results_type.from_raw_results(raw_results, data)
        return results
