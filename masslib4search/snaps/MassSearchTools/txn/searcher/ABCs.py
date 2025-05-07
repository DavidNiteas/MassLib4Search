from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar, Type, Dict, Tuple, Any

data_entity_type = TypeVar('data_entity_type', bound='SearchDataEntity')
config_entity_type = TypeVar('config_entity_type', bound='SearchConfigEntity')
results_entity_type = TypeVar('results_entity_type', bound='SearchResultsEntity')

class SearchDataEntity(BaseModel, ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    @abstractmethod
    def get_args(self) -> Tuple[Any,...]:
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
    def get_kwargs(self) -> Dict[str,Any]:
        '''Get the search function kwargs from the search config.'''
        pass
    
class SearchResultsEntity(BaseModel, ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    @classmethod
    @abstractmethod
    def from_raw_results(cls: Type[results_entity_type], raw_results) -> results_entity_type:
        '''Convert the raw search results to the search results entity.'''
        pass

class Searcher(BaseModel,ABC):
    
    # pydantic config
    model_config = dict(extra='forbid')
    
    # class variables
    results_type = SearchResultsEntity
    
    __slots__ = ['config']
    
    # instance variables
    config: SearchConfigEntity
    
    def check_data(self,data: SearchDataEntity):
        '''Check if the data is valid for the search.'''
        pass
    
    @abstractmethod
    def search_method(self, *args, **kwargs):
        '''
        The search function to be implemented by the searcher.
        The function should take the search data (*args) and config (**kwargs) as input.
        The function should return the raw search results, which will be converted to the search results entity by the SearchResultsEntity.from_raw_results() method
        '''
        pass
    
    def run(self, data: SearchDataEntity) -> SearchResultsEntity:
        '''Run the search function with the given data and config.'''
        self.check_data(data)
        raw_results = self.search_method(*data.get_args(), **self.config.get_kwargs())
        results = self.results_type.from_raw_results(raw_results)
        return results