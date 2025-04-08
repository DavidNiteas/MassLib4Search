import pickle
import rich.progress
import pandas as pd
from typing import List, Tuple, Dict, Union, Callable, Optional, Any, Literal, Hashable, Sequence

# dev import
import sys
sys.path.append('.')
from OpenDBUtils.OpenDBUtils import DBUtils

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
    
def save_df_to_SQLite(
    file_path: str,
    df: pd.DataFrame,
    table_name: str,
    chunk_size: int = 2048,
    max_workers: int = 8,
    table_replace: bool = True,
):
    db_utils = DBUtils(db_name=file_path, user='', password='', host='', port='',db_instance='sqlite')
    db_utils.store_df(
        df,
        table_name,
        chunk_size=chunk_size,
        max_workers=max_workers,
        table_replace=table_replace,
    )
    
def save_dfs_to_SQLite(
    file_path: str,
    dfs: Dict[str, pd.DataFrame],
    chunk_size: int = 2048,
    max_workers: int = 8,
    table_replace: bool = True,
):
    db_utils = DBUtils(db_name=file_path, user='', password='', host='', port='',db_instance='sqlite')
    db_utils.store_dict(
        dfs,
        chunk_size=chunk_size,
        max_workers=max_workers,
        table_replace=table_replace,
    )
    
def load_df_from_SQLite(
    file_path: str,
    table_name: str,
    chunk_size: int = 2048,
    max_workers: int = 8
) -> pd.DataFrame:
    db_utils = DBUtils(db_name=file_path, user='', password='', host='', port='',db_instance='sqlite')
    return db_utils.query_df(table_name, chunk_size=chunk_size, max_workers=max_workers)
    
def load_dfs_from_SQLite(
    file_path: str,
    chunk_size: int = 2048,
    max_workers: int = 8
) -> Dict[str, pd.DataFrame]:
    dfs = {}
    db_utils = DBUtils(db_name=file_path, user='', password='', host='', port='',db_instance='sqlite')
    for row in db_utils.execute_sql('PRAGMA table_list;'):
        table_name = row['name']
        dfs[table_name] = db_utils.query_df(table_name, chunk_size=chunk_size, max_workers=max_workers)
    return dfs

