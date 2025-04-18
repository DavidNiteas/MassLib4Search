from abc import ABC
import torch
from functools import partial
from typing import Callable, Optional

class EmbbedingSimilarityOperator(ABC):
    
    cpu_kwargs = {}
    cuda_kwargs = {}
    
    @classmethod
    def cuda_operator(
        cls,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        '''
        Returns a function that computes the similarity between two batches of embeddings.
        The function takes two batches of embeddings and returns a similarity matrix.
        The similarity matrix is a tensor of shape (batch_size_va, batch_size_vb) where each element (i, j)
        represents the similarity between the i-th embedding in the first batch and the j-th embedding
        in the second batch.
        The function should be able to handle batches of different sizes.
        '''
        raise NotImplementedError(f"{cls.__name__}.cuda_operator() not implemented")
    
    @classmethod
    def cpu_operator(
        cls,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        '''
        Returns a function that computes the similarity between two batches of embeddings.
        The function takes two batches of embeddings and returns a similarity matrix.
        The similarity matrix is a tensor of shape (batch_size_va, batch_size_vb) where each element (i, j)
        represents the similarity between the i-th embedding in the first batch and the j-th embedding
        in the second batch.
        The function should be able to handle batches of different sizes.
        '''
        raise NotImplementedError(f"{cls.__name__}.cpu_operator() not implemented")
    
    @classmethod
    def get_operator_kwargs(
        cls,
        device: torch.device,
        input_kwargs: Optional[dict] = None,
    ) -> dict:
        if device.type.startswith("cuda"):
            return {**cls.cuda_kwargs, **(input_kwargs or {})}
        else:
            return {**cls.cpu_kwargs, **(input_kwargs or {})}
        
    @classmethod
    def get_operator(
        cls, 
        device: torch.device,
        input_kwargs: Optional[dict] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if device.type.startswith("cuda"):
            return partial(cls.cuda_operator(), **cls.get_operator_kwargs(device,input_kwargs))
        else:
            return partial(cls.cpu_operator(), **cls.get_operator_kwargs(device,input_kwargs))
        
class SpectramSimilarityOperator(EmbbedingSimilarityOperator):
    
    dask_mode = None
    
    @classmethod
    def cuda_operator(
        cls,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        '''
        Returns a function that computes the similarity between two spectra.
        The function takes two batches of spectra and returns a similarity matrix.
        The similarity matrix is a zero-dimensional tensor
        '''
        raise NotImplementedError(f"{cls.__name__}.cuda_operator() not implemented")
    
    @classmethod
    def cpu_operator(
        cls,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        '''
        Returns a function that computes the similarity between two spectra.
        The function takes two batches of spectra and returns a similarity matrix.
        The similarity matrix is a zero-dimensional tensor
        '''
        raise NotImplementedError(f"{cls.__name__}.cpu_operator() not implemented")
    
    @classmethod
    def get_dask_mode(
        cls,
        input_dask_mode: Optional[str] = None,
    ) -> Optional[str]:
        if input_dask_mode is not None:
            return input_dask_mode
        else:
            return cls.dask_mode