from __future__ import annotations

from .lib_utils import io
from .search_utils import search_tools
from .base_lib import BaseLib
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