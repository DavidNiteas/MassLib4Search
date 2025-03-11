from __future__ import annotations
from . import base_tools,search_tools
from .mass_lib_utils import BaseLib
import dask
import dask.bag as db
import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

class AbsSpecLib(BaseLib):
    
    @staticmethod
    def lazy_init(
        peak_patterns: Union[List[NDArray[np.float64]]]
    ):
        pass