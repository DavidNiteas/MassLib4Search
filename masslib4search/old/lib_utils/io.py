import pickle
import rich.progress
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
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
    
def create_sqlite_engine(file_path: str) -> Engine:
    """创建SQLite数据库引擎"""
    return create_engine(f"sqlite:///{file_path}")
    
def save_df_to_sqlite(
    file_path: str,
    df: pd.DataFrame,
    table_name: str,
    chunk_size: int = 2048,
    if_exists: str = "replace",
    index: bool = False,
    dtype: Optional[Dict] = None
) -> None:
    """
    保存单个DataFrame到SQLite表
    
    :param file_path: SQLite数据库文件路径
    :param df: 要保存的DataFrame
    :param table_name: 目标表名
    :param chunk_size: 分块写入大小
    :param if_exists: 表存在时的处理策略（'fail', 'replace', 'append'）
    :param index: 是否保存索引
    :param dtype: 列数据类型字典（例如：{'price': Float(precision=2)}）
    """
    engine = create_sqlite_engine(file_path)
    try:
        with engine.begin() as connection:
            df.to_sql(
                name=table_name,
                con=connection,
                chunksize=chunk_size,
                if_exists=if_exists,
                index=index,
                dtype=dtype
            )
    except ValueError as e:
        raise ValueError(f"保存数据到表 {table_name} 失败: {str(e)}") from e
    finally:
        engine.dispose()
    
def save_dfs_to_sqlite(
    file_path: str,
    dfs: Dict[str, pd.DataFrame],
    chunk_size: int = 2048,
    if_exists: str = "replace",
    index: bool = False
) -> None:
    """
    批量保存多个DataFrame到SQLite
    
    :param file_path: SQLite数据库文件路径
    :param dfs: 字典格式的 {表名: DataFrame}
    :param chunk_size: 分块写入大小
    :param if_exists: 表存在时的处理策略
    :param index: 是否保存索引
    """
    engine = create_sqlite_engine(file_path)
    try:
        with engine.begin() as connection:
            for table_name, df in dfs.items():
                df.to_sql(
                    name=table_name,
                    con=connection,
                    chunksize=chunk_size,
                    if_exists=if_exists,
                    index=index
                )
    except ValueError as e:
        raise ValueError(f"批量保存数据失败: {str(e)}") from e
    finally:
        engine.dispose()
    
def load_df_from_sqlite(
    file_path: str,
    table_name: str,
    chunksize: Optional[int] = None,
    **read_sql_kwargs
) -> pd.DataFrame:
    """
    从SQLite表加载DataFrame
    
    :param file_path: SQLite数据库文件路径
    :param table_name: 要读取的表名
    :param chunksize: 分块读取大小（返回生成器）
    :return: DataFrame或生成器
    """
    engine = create_sqlite_engine(file_path)
    try:
        with engine.connect() as connection:
            return pd.read_sql_table(
                table_name=table_name,
                con=connection,
                chunksize=chunksize,
                **read_sql_kwargs
            )
    except SQLAlchemyError as e:
        raise ValueError(f"从表 {table_name} 读取数据失败: {str(e)}") from e
    finally:
        engine.dispose()
    
def load_dfs_from_sqlite(
    file_path: str,
    chunksize: Optional[int] = None,
    exclude_system_tables: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    加载数据库中所有表到DataFrame字典
    
    :param file_path: SQLite数据库文件路径
    :param chunksize: 分块读取大小
    :param exclude_system_tables: 是否排除SQLite系统表
    :return: {表名: DataFrame} 的字典
    """
    engine = create_sqlite_engine(file_path)
    dfs = {}
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if exclude_system_tables:
            tables = [t for t in tables if not t.startswith('sqlite_')]
            
        with engine.connect() as connection:
            for table in tables:
                dfs[table] = pd.read_sql_table(
                    table_name=table,
                    con=connection,
                    chunksize=chunksize
                )
        return dfs
    except SQLAlchemyError as e:
        raise ValueError(f"读取数据库失败: {str(e)}") from e
    finally:
        engine.dispose()

