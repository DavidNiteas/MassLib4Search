from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar, Type, Dict, Tuple, Any, Optional, List, Sequence

data_entity_type = TypeVar('data_entity_type', bound='SearchDataEntity')
config_entity_type = TypeVar('config_entity_type', bound='SearchConfigEntity')
results_entity_type = TypeVar('results_entity_type', bound='SearchResultsEntity')

class SearchDataEntity(BaseModel, ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    # instance variables
    __slots__ = ['query_id', 'ref_id']
    query_ids: Optional[List[Optional[Sequence]]] = None
    ref_ids: Optional[List[Optional[Sequence]] ]= None
    
    @abstractmethod
    def get_inputs(self) -> Tuple[
        Tuple[Any,...], # args
        Dict[str,Any], # kwargs
    ]:
        '''Get the search function input from the search data entity.'''
        pass
    
    @classmethod
    @abstractmethod
    def from_raw_data(cls: Type[data_entity_type], *args, **kwargs) -> data_entity_type:
        '''Convert the raw search data to the search data entity.'''
        pass
    
class SearchConfigEntity(BaseModel, ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    @abstractmethod
    def get_inputs(self) -> Tuple[
        Tuple[Any,...], # args
        Dict[str,Any], # kwargs
    ]:
        '''Get the search function kwargs from the search config.'''
        pass
    
class SearchResultsEntity(BaseModel, ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    @classmethod
    @abstractmethod
    def from_raw_results(
        cls: Type[results_entity_type], 
        raw_results, 
        data: SearchDataEntity,
    ) -> results_entity_type:
        '''Convert the raw search results to the search results entity.'''
        pass

class Searcher(BaseModel,ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    # class variables
    input_type = SearchDataEntity
    results_type = SearchResultsEntity
    
    # instance variables
    __slots__ = ['config']
    config: SearchConfigEntity
    
    def check_data(self,data: SearchDataEntity):
        '''Check if the data is valid for the search.'''
        if not isinstance(data, self.input_type):
            raise TypeError(f"Data must be of type {self.input_type.__name__}")
        
    def merge_inputs(
        self,
        data_args: Tuple[Any,...],
        data_kwargs: Dict[str,Any],
        config_args: Tuple[Any,...],
        config_kwargs: Dict[str,Any],
    ) -> Tuple[
        Tuple[Any,...], # args
        Dict[str,Any], # kwargs
    ]:
        args = []
        for data,config in zip(data_args,config_args):
            args.append(config if config is not None else data)
        kwargs = {**data_kwargs, **config_kwargs}
        return tuple(args), kwargs
    
    @abstractmethod
    def search_method(self, *args, **kwargs):
        '''
        The search function to be implemented by the searcher.
        The function should take the search data and config as input.
        The function should return the raw search results, which will be converted to the search results entity by the SearchResultsEntity.from_raw_results() method
        '''
        pass
    
    def run(self, data: SearchDataEntity) -> SearchResultsEntity:
        '''Run the search function with the given data and config.'''
        self.check_data(data)
        data_args, data_kwargs = data.get_inputs()
        config_args, config_kwargs = self.config.get_inputs()
        args, kwargs = self.merge_inputs(data_args, data_kwargs, config_args, config_kwargs)
        raw_results = self.search_method(*args, **kwargs)
        results = self.results_type.from_raw_results(raw_results,data)
        return results