from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Any,List

class ToolBox(BaseModel, ABC):
    
    '''
    `ToolBox`类是对工具函数组的再封装，目的是为了更好的管理工具函数，并提供统一的接口。\n
    类初始化时，需要传入工具函数组的配置参数，这些参数是工具函数的非数据性参数，比如`batch_size`，`num_workers`等。\n
    类需要实现两个接口，`run`和`run_by_queue`，`run`用于处理单组数据，`run_by_queue`用于处理队列数据。\n
    从设计上来说，`ToolBox`类是对工具函数群在功能类别上的抽象，**最好不要将不相关功能的函数放在一起**。\n
    除此之外，可以考虑将工具函数群本身当作类属性与类进行绑定，便于用户调用。
    '''
    
    # pydantic config
    model_config = ConfigDict(extra='forbid',slots=True,arbitrary_types_allowed=True)
    
    # instance variables
    
    @abstractmethod
    def run(self, input_data: List[Any], *args, **kwargs) -> Any:
        '''
        这个接口用于进行单组数据的处理，输入单组数据，输出单组数据的结果
        '''
        pass
    
    @abstractmethod
    def run_by_queue(self, input_data: List[List[Any]], *args, **kwargs) -> List[Any]:
        '''
        这个接口用于进行队列数据的处理，队列数据是**互不相关**的多组数据，输入结构化的队列数据，输出结构化的队列结果
        '''
        pass