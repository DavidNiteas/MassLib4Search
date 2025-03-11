import pickle
import rich.progress
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

def to_pickle_bytes(obj) -> bytes:
    return pickle.dumps(obj)

def from_pickle_bytes(b: bytes) -> Any:
    return pickle.loads(b)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pickle(
    path: str,
    desc: Optional[str] = None,
    **kwargs,
):
    if desc is None:
        desc = f"Loading data from {path}"
    with rich.progress.open(path, 'rb', description=desc, **kwargs) as f:
        return pickle.load(f)